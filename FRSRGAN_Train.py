import argparse
from math import log10

import gc
import pandas as pd
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import Dataset_OnlyHR
from FRVSR_Models import FRVSR

from FRVSR_Models import GeneratorLoss
from SRGAN.model import Generator, Discriminator
import SRGAN.pytorch_ssim as pts

parser = argparse.ArgumentParser(description='Train Super Resolution Models')

parser.add_argument('--num_epochs', default=1000, type=int, help='train epoch number')
parser.add_argument('--width', default=112, type=int, help='lr pic width')
parser.add_argument('--height', default=64, type=int, help='lr pic height')
parser.add_argument('--dataset_size', default=0, type=int, help='dataset_size, 0 to use all')
parser.add_argument('--batch_size', default=2, type=int, help='batch_size, default 2')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate, default 1e-5')
opt = parser.parse_args()

UPSCALE_FACTOR = 4
NUM_EPOCHS = opt.num_epochs
WIDTH = opt.width
HEIGHT = opt.height
batch_size = opt.batch_size
dataset_size = opt.dataset_size
lr = opt.lr

train_loader, val_loader = Dataset_OnlyHR.get_data_loaders(batch_size, dataset_size=dataset_size, validation_split=0.2)
num_train_batches = len(train_loader)
num_val_batches = len(val_loader)

netG = FRVSR(batch_size, lr_width=WIDTH, lr_height=HEIGHT)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters(), lr=lr)
optimizerD = optim.Adam(netD.parameters(), lr=lr)

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        fake_hrs = []
        fake_lrs = []
        fake_scrs = []
        real_scrs = []
        d_loss = 0
        netD.zero_grad()
        netG.init_hidden(device)

        for lr_img, hr_img in zip(data, target):
            # if torch.cuda.is_available():
            hr_img = hr_img.to(device)
            # if torch.cuda.is_available():
            lr_img = lr_img.to(device)

            fake_hr, fake_lr = netG(lr_img)

            real_out = netD(hr_img).mean()
            fake_out = netD(fake_hr).mean()

            fake_hrs.append(fake_hr)
            fake_lrs.append(fake_lr)
            fake_scrs.append(fake_out)
            real_scrs.append(real_out)

            d_loss += 1 - real_out + fake_out

        d_loss /= len(data)
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        g_loss = 0
        netG.zero_grad()
        idx = 0
        for fake_hr, fake_lr, fake_scr, hr_img, lr_img \
                in zip(fake_hrs, fake_lrs, fake_scrs, target, data):
            fake_hr = fake_hr.to(device)
            fake_lr = fake_lr.to(device)
            fake_scr = fake_scr.to(device)
            hr_img = hr_img.to(device)
            lr_img = lr_img.to(device)
            g_loss += generator_criterion(fake_scr, fake_hr, hr_img, fake_lr, lr_img, idx)
            idx += 1

        g_loss /= len(data)
        g_loss.backward()
        optimizerG.step()

        real_out = torch.Tensor(real_scrs).mean()
        fake_out = torch.Tensor(fake_scrs).mean()
        running_results['g_loss'] += g_loss.data.item() * batch_size
        running_results['d_loss'] += d_loss.data.item() * batch_size
        running_results['d_score'] += real_out.data.item() * batch_size
        running_results['g_score'] += fake_out.data.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))
        gc.collect()

    netG.eval()

    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    for val_lr, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size

        netG.init_hidden(device)

        batch_mse = []
        batch_ssim = []
        for lr, hr in zip(val_lr, val_hr):
            lr = lr.to(device)
            hr = hr.to(device)

            hr_est, lr_est = netG(lr)
            batch_mse.append(((hr_est - hr) ** 2).data.mean())
            batch_ssim.append(pts.ssim(hr_est, hr).item())

        batch_mse = torch.Tensor(batch_mse).mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = torch.Tensor(batch_ssim).mean()
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))
        gc.collect()

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % 1 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
