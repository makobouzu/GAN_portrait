import torch
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
import tqdm
from statistics import mean


# === 1. データの読み込み ===
# datasetrの準備
dataset = datasets.ImageFolder("data/",
    transform=transforms.Compose([
        transforms.ToTensor()
]))

batch_size = 64

# dataloaderの準備
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, nz=100, nch_g=128, nch=3):
        '''
        :param nz:    入力ベクトルｚの次元    100
        :param nch_g: 最終層の入力チャネル数  128
        :param nch:   出力画像のチャネル数    3(RGB)
        '''
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, nch_g*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nch_g*16),
            nn.ReLU(inplace=True),
            #(B, 100, 1, 1) -> (B, 1024(nch_g*16), 4, 4)

            nn.ConvTranspose2d(nch_g*16, nch_g*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nch_g*8),
            nn.ReLU(inplace=True),
            #(B, 1024(nch_g*16), 4, 4) -> (B, 512(nch_g*8), 8, 8)

            nn.ConvTranspose2d(nch_g*8, nch_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nch_g*4),
            nn.ReLU(inplace=True),
            #(B, 512(nch_g*8), 8, 8) -> (B, 256(nch_g*4), 16, 16)

            nn.ConvTranspose2d(nch_g*4, nch_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nch_g*2),
            nn.ReLU(inplace=True),
            #(B, 256(nch_g*4), 16, 16) -> (B, 128(nch_g*2), 32, 32)

            nn.ConvTranspose2d(nch_g*2, nch_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nch_g),
            nn.ReLU(inplace=True),
            #(B, 128(nch_g*2), 32, 32) -> (B, 64(nch_g), 64, 64)
            
            nn.ConvTranspose2d(nch_g, nch, 4, 2, 1, bias=False),
            nn.Tanh()
            #(B, 64(nch_g), 64, 64) -> (B, nch, 128, 128)
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, nch=3, nch_d=128):
        '''
        :param nch:   入力画像のチャネル数    3
        :param nch_d: 先頭層の出力チャネル数  128
        '''
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nch, nch_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #(B, 3(nch), 128, 128) -> (B, 64(nch_d), 64, 64)

            nn.Conv2d(nch_d, nch_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nch_d*2),
            nn.LeakyReLU(0.2, inplace=True),
            #(B, 64(nch_d), 64, 64) -> (B, 128(nch_d*2), 32, 32)

            nn.Conv2d(nch_d*2, nch_d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nch_d*4),
            nn.LeakyReLU(0.2, inplace=True),
            #(B, 128(nch_d*2), 32, 32) -> (B, 256(nch_d*4), 16, 16)

            nn.Conv2d(nch_d*4, nch_d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nch_d*8),
            nn.LeakyReLU(0.2, inplace=True),
            #(B, 256(nch_d*4), 16, 16) -> (B, 512(nch_d*8), 8, 8)

            nn.Conv2d(nch_d*8, nch_d*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nch_d*16),
            nn.LeakyReLU(0.2, inplace=True),
            #(B, 512(nch_d*8), 8, 8) -> (B, 1024(nch_d*16), 4, 4) 

            nn.Conv2d(nch_d*16, 1, 4, 1, 0, bias=False),
            #(B, 1024(nch_d*16), 4, 4) -> (B, 1, 1, 1)
        )

    def forward(self, x):
        return self.main(x).squeeze()


'''
(１) 潜在特徴100次元ベクトルzから、Generatorを使用して偽画像を生成する
(２) 偽画像をDiscriminatorで識別し、偽画像を実画像と騙せるようにGeneratorを学習する
(３) 実画像をDiscriminatorで識別する
(４) 偽画像を偽画像、実画像を実画像と識別できるようにDiscriminatorを学習する
'''

model_G = Generator().to("cuda:0")
model_D = Discriminator().to("cuda:0")

params_G = optim.Adam(model_G.parameters(),
    lr=0.0002, betas=(0.5, 0.999))
params_D = optim.Adam(model_D.parameters(),
    lr=0.0002, betas=(0.5, 0.999))

# 潜在特徴100次元ベクトルz
nz = 100

# ロスを計算するときのラベル変数
ones = torch.ones(batch_size).to("cuda:0") # 正例 1
zeros = torch.zeros(batch_size).to("cuda:0") # 負例 0
loss_f = nn.BCEWithLogitsLoss()

# 途中結果の確認用の潜在特徴z
#check_z = torch.randn(batch_size, nz, 1, 1).to("cuda:0")
check_z = torch.randn(1, nz, 1, 1).to("cuda:0")

# 訓練関数
def train_dcgan(model_G, model_D, params_G, params_D, data_loader):
    log_loss_G = []
    log_loss_D = []
    for real_img, _ in tqdm.tqdm(data_loader):
        batch_len = len(real_img)


        # == Generatorの訓練 ==
        # 偽画像を生成
        z = torch.randn(batch_len, nz, 1, 1).to("cuda:0")
        fake_img = model_G(z)

        # 偽画像の値を一時的に保存 => 無駄な計算（偽画像の生成）を省略する
        fake_img_tensor = fake_img.detach()

        # 偽画像を実画像（ラベル１）と騙せるようにロスを計算
        out = model_D(fake_img)
        loss_G = loss_f(out, ones[: batch_len])
        log_loss_G.append(loss_G.item())

        # 微分計算・重み更新 => Generator, Discriminatorの勾配を初期化してから、微分計算・重み更新
        model_D.zero_grad()
        model_G.zero_grad()
        loss_G.backward()
        params_G.step()


        # == Discriminatorの訓練 ==
        # sample_dataの実画像
        real_img = real_img.to("cuda:0")

        # 実画像を実画像（ラベル１）と識別できるようにロスを計算
        real_out = model_D(real_img)
        loss_D_real = loss_f(real_out, ones[: batch_len])

        # 一時的に保存しておいたTensorを使用して、無駄な計算（偽画像の生成）を省略する
        fake_img = fake_img_tensor

        # 偽画像を偽画像（ラベル０）と識別できるようにロスを計算
        fake_out = model_D(fake_img_tensor)
        loss_D_fake = loss_f(fake_out, zeros[: batch_len])

        # 実画像と偽画像のロスを合計
        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        # 微分計算・重み更新 => Generator, Discriminatorの勾配を初期化してから、微分計算・重み更新
        model_D.zero_grad()
        model_G.zero_grad()
        loss_D.backward()
        params_D.step()

    return mean(log_loss_G), mean(log_loss_D)

for epoch in range(300):
    train_dcgan(model_G, model_D, params_G, params_D, data_loader)

    # 訓練途中のモデル・生成画像の保存
    if epoch % 10 == 0:
        torch.save(
            model_G.state_dict(),
            "data/Weight_Generator/G_{:03d}.prm".format(epoch),
            pickle_protocol=4)
        torch.save(
            model_D.state_dict(),
            "data/Weight_Discriminator/D_{:03d}.prm".format(epoch),
            pickle_protocol=4)

    if epoch % 2 == 0:
        generated_img = model_G(check_z)
        save_image(generated_img,
                   "data/Generated_Image/{:03d}.jpg".format(epoch))