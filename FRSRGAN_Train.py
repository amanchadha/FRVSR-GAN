import argparse
from math import log10
import gc
import pandas as pd
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import Dataset_OnlyHR
import logger
from FRVSR_Models import FRVSR
from FRVSR_Models import GeneratorLoss
from SRGAN.model import Discriminator
import SRGAN.pytorch_ssim as pts

################################################## iSEEBETTER TRAINER KNOBS #############################################
UPSCALE_FACTOR = 4
########################################################################################################################

# Handle command line arguments
parser = argparse.ArgumentParser(description='Train iSeeBetter: Super Resolution Models')
parser.add_argument('-e', '--num_epochs', default=1000, type=int, help='train epoch number')
parser.add_argument('-w', '--width', default=112, type=int, help='lr pic width')
parser.add_argument('-ht', '--height', default=64, type=int, help='lr pic height')
parser.add_argument('-d', '--dataset_size', default=0, type=int, help='dataset_size, 0 to use all')
parser.add_argument('-b', '--batch_size', default=2, type=int, help='batch_size, default 2')
parser.add_argument('-l', '--lr', default=1e-5, type=float, help='learning rate, default 1e-5')
parser.add_argument('-v', '--debug', default=False, action='store_true', help='Print debug spew.')

args = parser.parse_args()
NUM_EPOCHS = args.num_epochs
WIDTH = args.width
HEIGHT = args.height
batch_size = args.batch_size
dataset_size = args.dataset_size
lr = args.lr

# Load dataset
trainLoader, valLoader = Dataset_OnlyHR.get_data_loaders(batch_size, dataset_size=dataset_size, validation_split=0.2)
numTrainBatches = len(trainLoader)
numValBatches = len(valLoader)

# Initialize Logger
logger.initLogger(args.debug)

# Use Generator as FRVSR
netG = FRVSR(batch_size, lr_width=WIDTH, lr_height=HEIGHT)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

# Use Discriminator from SRGAN
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generatorCriterion = GeneratorLoss()

if torch.cuda.is_available():
    def printCUDAStats():
        logger.info("# of CUDA devices detected:", torch.cuda.device_count())
        logger.info("Using CUDA device #:", torch.cuda.current_device())
        logger.info("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    printCUDAStats()

    netG.cuda()
    netD.cuda()
    generatorCriterion.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use Adam optimizer
optimizerG = optim.Adam(netG.parameters(), lr=lr)
optimizerD = optim.Adam(netD.parameters(), lr=lr)

results = {'DLoss': [], 'GLoss': [], 'DScore': [], 'GScore': [], 'PSNR': [], 'SSIM': []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(trainLoader)
    runningResults = {'batchSize': 0, 'DLoss': 0, 'GLoss': 0, 'DScore': 0, 'GScore': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:
        GUpdateFirst = True
        batch_size = data.size(0)
        runningResults['batchSize'] += batch_size

        ################################################################################################################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ################################################################################################################
        fakeHRs = []
        fakeLRs = []
        fakeScrs = []
        realScrs = []
        DLoss = 0

        # Zero-out gradients, i.e., start afresh
        netD.zero_grad()
        
        netG.init_hidden(device)

        for LRImg, HRImg in zip(data, target):
            # if torch.cuda.is_available():
            HRImg = HRImg.to(device)
            # if torch.cuda.is_available():
            LRImg = LRImg.to(device)

            fakeHR, fakeLR = netG(LRImg)

            realOut = netD(HRImg).mean()
            fake_out = netD(fakeHR).mean()

            fakeHRs.append(fakeHR)
            fakeLRs.append(fakeLR)
            fakeScrs.append(fake_out)
            realScrs.append(realOut)

            DLoss += 1 - realOut + fake_out

        DLoss /= len(data)

        # Calculate gradients
        DLoss.backward(retain_graph=True)

        # Update weights
        optimizerD.step()

        ################################################################################################################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ################################################################################################################
        GLoss = 0

        # Zero-out gradients, i.e., start afresh
        netG.zero_grad()

        idx = 0
        for fakeHR, fakeLR, fake_scr, HRImg, LRImg in zip(fakeHRs, fakeLRs, fakeScrs, target, data):
            fakeHR = fakeHR.to(device)
            fakeLR = fakeLR.to(device)
            fake_scr = fake_scr.to(device)
            HRImg = HRImg.to(device)
            LRImg = LRImg.to(device)
            GLoss += generatorCriterion(fake_scr, fakeHR, HRImg, fakeLR, LRImg, idx)
            idx += 1

        GLoss /= len(data)

        # Calculate gradients
        GLoss.backward()

        # Update weights
        optimizerG.step()

        realOut = torch.Tensor(realScrs).mean()
        fake_out = torch.Tensor(fakeScrs).mean()
        runningResults['GLoss'] += GLoss.data.item() * batch_size
        runningResults['DLoss'] += DLoss.data.item() * batch_size
        runningResults['DScore'] += realOut.data.item() * batch_size
        runningResults['GScore'] += fake_out.data.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, runningResults['DLoss'] / runningResults['batchSize'],
            runningResults['GLoss'] / runningResults['batchSize'],
            runningResults['DScore'] / runningResults['batchSize'],
            runningResults['GScore'] / runningResults['batchSize']))
        gc.collect()

    netG.eval()

    val_bar = tqdm(valLoader)
    validatingResults = {'MSE': 0, 'SSIMs': 0, 'PSNR': 0, 'SSIM': 0, 'batchSize': 0}
    val_images = []
    for val_lr, val_hr in val_bar:
        batch_size = val_lr.size(0)
        validatingResults['batchSize'] += batch_size

        netG.init_hidden(device)

        batchMSE = []
        batchSSIM = []
        for lr, hr in zip(val_lr, val_hr):
            lr = lr.to(device)
            hr = hr.to(device)

            HREst, LREst = netG(lr)
            batchMSE.append(((HREst - hr) ** 2).data.mean())
            batchSSIM.append(pts.SSIM(HREst, hr).item())

        batchMSE = torch.Tensor(batchMSE).mean()
        validatingResults['MSE'] += batchMSE * batch_size
        batchSSIM = torch.Tensor(batchSSIM).mean()
        validatingResults['SSIMs'] += batchSSIM * batch_size
        validatingResults['PSNR'] = 10 * log10(1 / (validatingResults['MSE'] / validatingResults['batchSize']))
        validatingResults['SSIM'] = validatingResults['SSIMs'] / validatingResults['batchSize']
        val_bar.set_description(
            desc='[Converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' %
                 (validatingResults['PSNR'], validatingResults['SSIM']))
        gc.collect()

    # Save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

    # Save Loss\Scores\PSNR\SSIM
    results['DLoss'].append(runningResults['DLoss'] / runningResults['batchSize'])
    results['GLoss'].append(runningResults['GLoss'] / runningResults['batchSize'])
    results['DScore'].append(runningResults['DScore'] / runningResults['batchSize'])
    results['GScore'].append(runningResults['GScore'] / runningResults['batchSize'])
    results['PSNR'].append(validatingResults['PSNR'])
    results['SSIM'].append(validatingResults['SSIM'])

    if epoch % 1 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(data={'Loss_D': results['DLoss'], 'Loss_G': results['GLoss'], 'Score_D': results['DScore'],
                                  'Score_G': results['GScore'], 'PSNR': results['PSNR'], 'SSIM': results['SSIM']},
                                  index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
