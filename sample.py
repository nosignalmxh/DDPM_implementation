# 本代码用于根据训练好的unet模型进行diffusion过程从而生成图像

# 导入需要的包以及指定使用的gpu
import  os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torchvision
import torch.optim as optim
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from model import unet

# 参数设置
bz = 8
T = 1000
dropout_p = 0.1
channel = 128
mult = [1, 2, 3, 4]
att_num = [2]
res_num = 2
load_path = './checkpoints/ckpt_240_ema.pt'
beta_1 = 1e-4
beta_2 = 0.02
noise_save_name = 'noise.png'
betas = torch.linspace(beta_1, beta_2, T).double()
alphas = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)
al_hat_2 = torch.sqrt(1 - alpha_hat)
nrow = 1
alphas_bar_prev = F.pad(alpha_hat, [1, 0], value=1)[:T]
coeff1 = torch.sqrt(1. / alphas)
coeff2 = coeff1 * (1. - alphas) / al_hat_2
posterior_var = betas * (1. - alphas_bar_prev) / (1. - alpha_hat)

# 从输入张量v中抽取出特定索引的元素，并返回这些元素形成的新的张量，这个新的张量具有特定的形状。
def extract(v, t, x_shape):
    device = t.device
    v = v.to(device)  
    out = torch.gather(v, index=t, dim=0).float() # torch.gather()函数是用于从输入张量v中在第一个维度上抽取出特定索引的元素。这些索引由t提供
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)) # 用于调整张量的形状。新的形状由t的第一个维度大小决定，并且根据x_shape的长度来增加额外的维度，这些额外的维度大小为1。
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet(T, channel, mult, att_num, res_num, dropout_p).to(device)
    model.load_state_dict(torch.load(load_path, map_location=device))
    print("model load weight done.")
    model.eval()
    # 随机选择高斯噪声并且保存
    noisyImage = torch.randn([bz, 3, 32, 32], device=device)
    saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
    save_image(saveNoisy, noise_save_name, nrow=nrow)
    
    with torch.no_grad(): 
        x_t = noisyImage
        for time_step in tqdm(reversed(range(T)), desc="Sampling progress"):
            t = (x_t.new_ones([noisyImage.shape[0], ], dtype=torch.long) * time_step).to(device)
            var = torch.cat([posterior_var[1:2], betas[1:]])
            var = extract(var, t, x_t.shape)
            eps = model(x_t, t)
            xt_prev_mean = (extract(coeff1, t, x_t.shape) * x_t - extract(coeff2, t, x_t.shape) * eps)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = xt_prev_mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        
            if time_step % 100 == 0:
                sampledImgs = torch.clamp(x_t, -1, 1)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                save_image(sampledImgs, f"image_{time_step}.png", nrow=nrow)
    
if __name__ == '__main__':
    main()