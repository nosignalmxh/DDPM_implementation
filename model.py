#本代码构建了unet模型用于噪声预测，在默认参数(ch_mult = [1, 2, 3, 4] 且 num_res_blocks = 2)下的结构为：
'''
1. **输入层：** 
   - 输入：`(bz, 3, H, W)`

2. **时间嵌入层：**
   - 输入：`(bz,)`
   - 输出：`(bz, ch * 4)`

3. **卷积层 `self.head`：**
   - 输入：`(bz, 3, H, W)`
   - 输出：`(bz, ch, H, W)`

4. **下采样路径：** 
   - 第一阶段：
     - 输入：`(bz, ch, H, W)`
     - 经过2个残差块后，输出：`(bz, ch * ch_mult[0], H, W)`
     - 经过下采样层后，输出：`(bz, ch * ch_mult[0], H/2, W/2)`
   - 第二阶段：
     - 输入：`(bz, ch * ch_mult[0], H/2, W/2)`
     - 经过2个残差块后，输出：`(bz, ch * ch_mult[1], H/2, W/2)`
     - 经过下采样层后，输出：`(bz, ch * ch_mult[1], H/4, W/4)`
   - 第三阶段：
     - 输入：`(bz, ch * ch_mult[1], H/4, W/4)`
     - 经过2个残差块后，输出：`(bz, ch * ch_mult[2], H/4, W/4)`
     - 经过下采样层后，输出：`(bz, ch * ch_mult[2], H/8, W/8)`
   - 第四阶段：
     - 输入：`(bz, ch * ch_mult[2], H/8, W/8)`
     - 经过2个残差块后，输出：`(bz, ch * ch_mult[3], H/8, W/8)`

5. **中间层：**
   - 输入：`(bz, ch * ch_mult[3], H/8, W/8)`
   - 经过2个残差块后，输出：`(bz, ch * ch_mult[3], H/8, W/8)`

6. **上采样路径：**
   - 第一阶段：
     - 输入：`(bz, ch * ch_mult[3], H/8, W/8)`
     - 拼接后输入：`(bz, ch * ch_mult[3] + ch * ch_mult[2], H/8, W/8)`
     - 经过2个残差块后，输出：`(bz, ch * ch_mult[3], H/8, W/8)`
     - 经过上采样层后，输出：`(bz, ch * ch_mult[3], H/4, W/4)`
   - 第二阶段：
     - 输入：`(bz, ch * ch_mult[3], H/4, W/4)`
     - 拼接后输入：`(bz, ch * ch_mult[3] + ch * ch_mult[1], H/4, W/4)`
     - 经过2个残差块后，输出：`(bz, ch * ch_mult[2], H/4, W/4)`
     - 经过上采样层后，输出：`(bz, ch * ch_mult[2], H/2, W/2)`
   - 第三阶段：
     - 输入：`(bz, ch * ch_mult[2], H/2, W/2)`
     - 拼接后输入：`(bz, ch * ch_mult[2] + ch * ch_mult[0], H/2, W/2)`
     - 经过2个残差块后，输出：`(bz, ch * ch_mult[1], H/2, W/2)`
     - 经过上采样层后，输出：`(bz, ch * ch_mult[1], H, W)`
   - 第四阶段：
     - 输入：`(bz, ch * ch_mult[1], H, W)`
     - 拼接后输入：`(bz, ch * ch_mult[1] + ch, H, W)`
     - 经过2个残差块后，输出：`(bz, ch, H, W)`

7. **输出层 `self.tail`：**
   - 输入：`(bz, ch, H, W)`
   - 输出：`(bz, 3, H, W)`

'''
   
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

# [x] => [|x|]
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# =>[T,dim]
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.true_divide(torch.arange(0, d_model, step=2), d_model) * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

# [bz,3,2x,2x] => [bz,3,x,x]
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)

    def forward(self, x, temb):
        x = self.main(x)
        return x

# [bz,3,x,x] => [bz,3,2x,2x]
class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

#[b,c,h,w]=>[b,c,h,w]  放大特征
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x) #[b,c,h,w]=>[b,c,h,w]
        q = self.proj_q(h) #[b,c,h,w]=>[b,c,h,w]
        k = self.proj_k(h) #[b,c,h,w]=>[b,c,h,w]
        v = self.proj_v(h) #[b,c,h,w]=>[b,c,h,w]

        q = q.permute(0, 2, 3, 1).view(B, H * W, C) #[b,c,h,w]=>[b,h*w,c]
        k = k.view(B, C, H * W) #[b,c,h,w]=>[b,c,h*w]
        w = torch.bmm(q, k) * (int(C) ** (-0.5)) #[b,c,h,w]=>[b,h*w,h*w]
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1) #[b,h*w,h*w]=>[b,h*w,h*w]

        v = v.permute(0, 2, 3, 1).view(B, H * W, C) #[b,c,h,w]=>[b,h*w,c]
        h = torch.bmm(w, v) #[b,h*w,h*w] [b,c,h,w] => [b,h*w,c]
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2) # [b,h*w,c] => [b,c,h,w]
        h = self.proj(h) # [b,c,h,w] => [b,c,h,w]
        x = x.contiguous(memory_format=torch.channels_last)
        h = h.contiguous(memory_format=torch.channels_last)

        return x + h

# [b,out_ch,h,w]=>[b,out_ch,h,w]
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch: # 是否shortcut
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn: # 是否加入attention模块
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x) # [b,out_ch,h,w]=>[b,out_ch,h,w]
        h += self.temb_proj(temb)[:, :, None, None] # [b,out_ch,h,w] + (B, out_ch)=> [b,out_ch,h,w]
        h = self.block2(h)# [b,out_ch,h,w]=>[b,out_ch,h,w]

        h = h + self.shortcut(x)# [b,out_ch,h,w]=>[b,out_ch,h,w]
        h = self.attn(h)# [b,out_ch,h,w]=>[b,out_ch,h,w]
        return h


class unet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        
    def forward(self, x, t):
        temb = self.time_embedding(t) #[bz,]=>[bz,tdim]
        # Downsampling
        h = self.head(x) # [bz, 3, H, W]=>[bz, channel, H, W]
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)  # 通过每个下采样残差块，输出形状可能变为 (bz, now_ch, H', W')
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)  # 通过中间的残差块，输出形状不变，仍然为 (bz, now_ch, H', W')
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1) # 从 hs 中取出对应的特征图并与 h 拼接，输出形状为 (bz, now_ch*2, H', W')
            h = layer(h, temb) # 通过每个上采样残差块，输出形状可能变为 (bz, now_ch, H'', W'')，如果有上采样操作则 H'', W'' 会增大
        h = self.tail(h) # 输出形状为 (bz, 3, H, W)

        assert len(hs) == 0
        return h