"""
This file trains FRVSR on a single low resolution video source and upscales it to 4x.
aman@amanchadha.com

Adapted from FR-SRGAN, MIT 6.819 Advances in Computer Vision, Nov 2018
"""

import gc
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import pandas as pd

from SRGAN import pytorch_ssim

torch.backends.cudnn.benchmark = True
import AFRVSRModels
import Dataset_OnlyHR


def load_model(model_name, batch_size, width, height):
    model = AFRVSRModels.FRVSR(batch_size=batch_size, lr_height=height, lr_width=width)
    if model_name != '':
        model_path = f'./models/{model_name}'
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    return model

def run():
    # Parameters
    num_epochs = 25
    output_period = 10
    batch_size = 4
    width, height = 112, 64

    epoch_train_losses = []
    epoch_valid_losses = []

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model('', batch_size, width, height)
    model = model.to(device)
    
    torch.save(model.state_dict(), "models/FRVSRTest")

    train_loader, val_loader = Dataset_OnlyHR.get_data_loaders(batch_size, dataset_size=0, validation_split=0.2)
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    flow_criterion = nn.MSELoss().to(device)
    content_criterion = AFRVSRModels.Loss().to(device)

    ssim_loss = pytorch_ssim.SSIM(window_size=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    epoch = 1
    while epoch <= num_epochs:
        epoch_train_loss = 0.0
        epoch_valid_loss = 0.0
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        print(train_loader)

        for batch_num, (lr_imgs, hr_imgs) in enumerate(train_loader, 1):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            # print(f'hrimgs.shape is {hr_imgs.shape}')
            # print(f'lrimgs.shape is {lr_imgs.shape}')
            optimizer.zero_grad()
            model.init_hidden(device)
            batch_content_loss = 0
            batch_flow_loss = 0

            # lr_imgs = 7 * 4 * 3 * H * W
            cnt = 0
            for lr_img, hr_img in zip(lr_imgs, hr_imgs):
                # print(lr_img.shape)
                hr_est, lr_est = model(lr_img)
                content_loss = content_criterion(hr_est, hr_img)
                flow_loss = torch.mean((lr_img - lr_est) ** 2)
                # flow_loss = ssim_loss(lr_img, lr_est)
                #print(f'content_loss is {content_loss}, flow_loss is {flow_loss}')
                batch_content_loss += content_loss
                if cnt > 0:
                    batch_flow_loss += flow_loss
                cnt += 1

            #print(f'loss is {loss}')
            loss = batch_content_loss + batch_flow_loss
            loss.backward()

            # dot = get_dot()
            # dot.save('tmp.dot')
            print(torch.max(model.fnet.out.grad))
            print(torch.max(model.EstHrImg.grad))
            print(f'content_loss {batch_content_loss}, flow_loss {batch_flow_loss}')
            
            # print("success")
            optimizer.step()
            running_loss += loss.item()
            epoch_train_loss = (epoch_train_loss * (batch_num - 1) + loss.item()) / batch_num

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num * 1.0 / num_train_batches,
                    running_loss / output_period
                ), file=sys.stderr)
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/FRVSR.%d" % epoch)

        model.eval()
        with torch.no_grad():
            output_period = 0
            running_loss = 0
            for batch_num, (lr_imgs, hr_imgs) in enumerate(val_loader, 1):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                model.init_hidden(device)
                batch_content_loss = 0
                batch_flow_loss = 0

                # lr_imgs = 7 * 4 * 3 * H * W
                cnt = 0
                for lr_img, hr_img in zip(lr_imgs, hr_imgs):
                    # print(lr_img.shape)
                    hr_est, lr_est = model(lr_img)
                    content_loss = content_criterion(hr_est, hr_img)
                    flow_loss = torch.mean((lr_img - lr_est) ** 2)
                    # flow_loss = ssim_loss(lr_img, lr_est)
                    # print(f'content_loss is {content_loss}, flow_loss is {flow_loss}')
                    batch_content_loss += content_loss
                    if cnt > 0:
                        batch_flow_loss += flow_loss
                    cnt += 1
                output_period += 1
                loss = batch_content_loss + batch_flow_loss
                running_loss += loss
                epoch_valid_loss = (epoch_valid_loss * (batch_num - 1) + loss) / batch_num

            print('Epoch: [%d], Average loss: %.3f' % (epoch, running_loss / output_period), file=sys.stderr)

        gc.collect()

        epoch_train_losses.append(epoch_train_loss)
        epoch_valid_losses.append(epoch_valid_loss)

        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'train_Loss': epoch_train_losses, 'valid_Loss': epoch_valid_losses},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'FRVSR_' + str(4) + '_train_results.csv', index_label='Epoch')

        epoch += 1


if __name__ == "__main__":
    print('Starting training')
    run()
    print('Training terminated')
