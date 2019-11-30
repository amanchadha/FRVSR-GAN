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
parser.add_argument('-b', '--batchSize', default=2, type=int, help='batchSize, default 2')
parser.add_argument('-l', '--lr', default=1e-5, type=float, help='learning rate, default 1e-5')
parser.add_argument('-x', '--express', default=False, action='store_true', help='Express mode: no validation.')
parser.add_argument('-v', '--debug', default=False, action='store_true', help='Print debug spew.')

args = parser.parse_args()
NUM_EPOCHS = args.num_epochs
WIDTH = args.width
HEIGHT = args.height
batchSize = args.batchSize
dataset_size = args.dataset_size
lr = args.lr
express = args.express

# Load dataset
trainLoader, valLoader = Dataset_OnlyHR.get_data_loaders(batchSize, dataset_size=dataset_size, validation_split=0.1)
numTrainBatches = len(trainLoader)
numValBatches = len(valLoader)

# Initialize Logger
logger.initLogger(args.debug)

# Use Generator as FRVSR
netG = FRVSR(batchSize, lr_width=WIDTH, lr_height=HEIGHT)
print('# of Generator parameters:', sum(param.numel() for param in netG.parameters()))

# Use Discriminator from SRGAN
netD = Discriminator()
print('# of Discriminator parameters:', sum(param.numel() for param in netD.parameters()))

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

def trainModel():
    trainBar = tqdm(trainLoader)
    runningResults = {'batchSize': 0, 'DLoss': 0, 'GLoss': 0, 'DScore': 0, 'GScore': 0}

    netG.train()
    netD.train()

    for data, target in trainBar:
        batchSize = data.size(0)
        runningResults['batchSize'] += batchSize

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
            HRImg = HRImg.to(device)
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
        runningResults['GLoss'] += GLoss.data.item() * batchSize
        runningResults['DLoss'] += DLoss.data.item() * batchSize
        runningResults['DScore'] += realOut.data.item() * batchSize
        runningResults['GScore'] += fake_out.data.item() * batchSize

        trainBar.set_description(desc='[Epoch: %d/%d] D Loss: %.4f G Loss: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, runningResults['DLoss'] / runningResults['batchSize'],
            runningResults['GLoss'] / runningResults['batchSize'],
            runningResults['DScore'] / runningResults['batchSize'],
            runningResults['GScore'] / runningResults['batchSize']))
        gc.collect()

    netG.eval()

    return runningResults

def validateModel():
    validationBar = tqdm(valLoader)
    validationResults = {'MSE': 0, 'SSIMs': 0, 'PSNR': 0, 'SSIM': 0, 'batchSize': 0}
    for valLR, valHR in validationBar:
        batchSize = valLR.size(0)
        validationResults['batchSize'] += batchSize

        netG.init_hidden(device)

        batchMSE = []
        batchSSIM = []
        for lr, hr in zip(valLR, valHR):
            lr = lr.to(device)
            hr = hr.to(device)

            HREst, LREst = netG(lr)
            batchMSE.append(((HREst - hr) ** 2).data.mean())
            batchSSIM.append(pts.SSIM(HREst, hr).item())

        batchMSE = torch.Tensor(batchMSE).mean()
        validationResults['MSE'] += batchMSE * batchSize
        batchSSIM = torch.Tensor(batchSSIM).mean()
        validationResults['SSIMs'] += batchSSIM * batchSize
        validationResults['PSNR'] = 10 * log10(1 / (validationResults['MSE'] / validationResults['batchSize']))
        validationResults['SSIM'] = validationResults['SSIMs'] / validationResults['batchSize']
        validationBar.set_description(desc='[Converting LR images to SR images] PSNR: %.4fdB SSIM: %.4f' %
                                      (validationResults['PSNR'], validationResults['SSIM']))
        gc.collect()

        return validationResults

def saveModelParams(epoch, runningResults, validationResults={}):
    # Save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

    # Save Loss\Scores\PSNR\SSIM
    results['DLoss'].append(runningResults['DLoss'] / runningResults['batchSize'])
    results['GLoss'].append(runningResults['GLoss'] / runningResults['batchSize'])
    results['DScore'].append(runningResults['DScore'] / runningResults['batchSize'])
    results['GScore'].append(runningResults['GScore'] / runningResults['batchSize'])
    #results['PSNR'].append(validationResults['PSNR'])
    #results['SSIM'].append(validationResults['SSIM'])

    if epoch % 1 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(data={'DLoss': results['DLoss'], 'GLoss': results['GLoss'], 'DScore': results['DScore'],
                                  'GScore': results['GScore']},#, 'PSNR': results['PSNR'], 'SSIM': results['SSIM']},
                                  index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

def main():
    """ Lets begin the training process! """

    for epoch in range(1, NUM_EPOCHS + 1):
        runningResults = trainModel()

        # Do validation only if express mode is not enabled
        if not express:
            validationResults = validateModel()

        saveModelParams(epoch, runningResults)#, validationResults)

if __name__ == "__main__":
    main()
