# 本代码用于训练diffusion模型

# 导入需要的包以及指定使用的gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torchvision
import torch.optim as optim
import tqdm
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from model import unet
import copy

# 参数设置
bz = 128
T = 1000
dropout_p = 0.1
channel = 128
mult = [1, 2, 3, 4]
att_num = [2]
res_num = 2
from_pretrained = None
pretrained_path = './checkpoints/'
save_weight_dir = './checkpoints/'
epoch = 1280
lr = 2e-4
beta_1 = 1e-4
beta_2 = 0.02
grad_clip = 1.
ema_decay = 0.9999

betas = torch.linspace(beta_1, beta_2, T).double()
alphas = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)
sqrt_al_hat = torch.sqrt(alpha_hat)
al_hat_2 = torch.sqrt(1 - alpha_hat)

# 从输入张量v中抽取出特定索引的元素，并返回这些元素形成的新的张量，这个新的张量具有特定的形状。
def extract(v, t, x_shape):
    device = t.device
    v = v.to(device) 
    out = torch.gather(v, index=t, dim=0).float() # torch.gather()函数是用于从输入张量v中在第一个维度上抽取出特定索引的元素。这些索引由t提供
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)) # 用于调整张量的形状。新的形状由t的第一个维度大小决定，并且根据x_shape的长度来增加额外的维度，这些额外的维度大小为1。

def main():
    # 设置数据集和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading the data...")
    start_time = time.time()
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bz, shuffle=True, num_workers=4)
    end_time = time.time()
    print("Data loaded in ", end_time - start_time, " seconds")
    print("Initializing the model...")
    model = unet(T, channel, mult, att_num, res_num, dropout_p)
    print("Model initialized. Moving model to device...")
    model = model.to(device)
    print("Model moved to device.")
    # Initialize EMA model
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    # 从记录点恢复训练
    if from_pretrained:
        print("Loading model from pretrained weights...")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("Loaded model from pretrained weights.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print("Starting the training loop...")
    for e in range(epoch):
        print(f"Starting epoch {e+1}")
        start_time = time.time()
        trainloader_tqdm = tqdm.tqdm(trainloader, desc=f'Epoch {e+1}/{epoch}')
        for images, labels in trainloader_tqdm:
            optimizer.zero_grad()
            x_0 = images.to(device)
            t = torch.randint(T, size=(x_0.shape[0], ), device=x_0.device) # 随机时间
            noise = torch.randn_like(x_0) # 随机噪声
            x_t = (extract(sqrt_al_hat, t, x_0.shape) * x_0 + extract(al_hat_2, t, x_0.shape) * noise)
            loss = F.mse_loss(model(x_t, t), noise, reduction='none').sum() / 1000.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # Update EMA parameters
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_((1 - ema_decay) * param.data)
            trainloader_tqdm.set_postfix({'loss': f'{loss.item():.4f}'})
        torch.save(ema_model.state_dict(), os.path.join(save_weight_dir, 'ckpt_' + str(e) + "_ema.pt")) # Save the EMA model
        end_time = time.time()
        print("Epoch finished in ", end_time - start_time, " seconds")
    print("Training finished.")

    
if __name__ == '__main__':
    main()