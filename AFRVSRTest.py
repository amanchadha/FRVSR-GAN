"""
This file does a quick check of a trained FRVSR model on a single low resolution video source and upscales it to 4x.
aman@amanchadha.com

Adapted from FR-SRGAN, MIT 6.819 Advances in Computer Vision, Nov 2018
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt
import Dataset_OnlyHR
import AFRVSRModels
from skimage import img_as_ubyte
from skimage.util import img_as_float32

# TODO
verbose = 0

def trunc(tensor):
    # tensor = tensor.clone()
    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1
    return tensor

def test_optic_flow(frame1, frame2):
    # im1 = img_as_ubyte(frame1)
    # im2 = img_as_ubyte(frame2)
    im1 = cv2.imread('im1.png')
    im2 = cv2.imread('im2.png')
    frame1 = img_as_float32(im1)
    frame2 = img_as_float32(im2)
    prvs = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    flow[..., 0] /= flow.shape[1] / 2
    flow[..., 1] /= flow.shape[0] / 2
    flow *= -1
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            flow[i, j, 0] += (j / flow.shape[1] * 2 - 1)
            flow[i, j, 1] += (i / flow.shape[0] * 2 - 1)

    print(flow.shape)

    torch_frame1 = torch.unsqueeze(torch.tensor(frame1).permute(2, 0, 1), 0)

    # print(frame1.shape)
    # print(torch_frame1.shape)
    # print(torch_frame1)
    flow = flow.astype(np.float32, copy=False)
    est_frame2 = func.grid_sample(torch_frame1, torch.unsqueeze(torch.tensor(flow), 0))
    res_img = img_as_ubyte(est_frame2[0].permute(1, 2, 0).numpy())
    cv2.imwrite('est_frame2.png', res_img)
    # flow_len = np.expand_dims(np.sqrt((flow[...,0]**2 + flow[...,1]**2)), 2)
    # flow /= flow_len
    # print(flow)
    pass
    exit(0)
    # cv2.imshow('frame2', rgb)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #     elif k == ord('s'):
    #         cv2.imwrite('opticalmyhsv.pgm', rgb)
    #
    # cap.release()
    # cv2.destroyAllWindows()


import math


def psnr(img1, img2):
    # print(img1.size())
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Single Video')
    parser.add_argument('--model', default='./epochs/netG_epoch_4_7.pth', type=str, help='AFRVSR Model')
    opt = parser.parse_args()

    UPSCALE_FACTOR = 4
    MODEL_NAME = opt.model

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AFRVSRModels.FRVSR(0, 0, 0)
        model.to(device)

        # for cpu
        # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
        checkpoint = torch.load(MODEL_NAME, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()

        train_loader, val_loader = Dataset_OnlyHR.get_data_loaders(batch=1, fixedIndices=0, dataset_size=0, validation_split=1, shuffle_dataset=True)

        tot_psnr = 0
        for idx, (lr_example, hr_example) in enumerate(val_loader, 1):
            out_psnr = 0
            fps = 6
            frame_numbers = 7
            # frame_numbers = 100
            lr_width = lr_example.shape[4]
            lr_height = lr_example.shape[3]
            model.set_param(batch_size=1, width=lr_width, height=lr_height)
            model.init_hidden(device)

            hr_video_size = (lr_width * UPSCALE_FACTOR, lr_height * UPSCALE_FACTOR)
            lr_video_size = (lr_width, lr_height)

            output_sr_name = 'AFRVSROut_' + str(UPSCALE_FACTOR) + f'_{idx}_' + 'Random_Sample.mp4'
            output_gt_name = 'AFRVSROut_' + 'GroundTruth' + f'_{idx}_' + 'Random_Sample.mp4'
            output_lr_name = 'AFRVSROut_' + 'LowRes' + '_' + 'Random_Sample.mp4'
            output_aw_name = 'AFRVSROut_' + 'IntermediateWarp' + '_' + 'Random_Sample.mp4'

            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            hr_video_writer = cv2.VideoWriter(output_sr_name, fourcc, fps, hr_video_size)
            lr_video_writer = cv2.VideoWriter(output_lr_name, fourcc, fps, lr_video_size)
            aw_video_writer = cv2.VideoWriter(output_aw_name, fourcc, fps, hr_video_size)
            gt_video_writer = cv2.VideoWriter(output_gt_name, fourcc, fps, hr_video_size)
            # read frame
            # test_optic_flow(lr_example[0][0].permute(1,2,0).numpy(), \
            #                  lr_example[1][0].permute(1,2,0).numpy())

            for image, truth in zip(lr_example, hr_example):
                # plt.subplot(121)
                # plt.imshow(image[0].permute(1,2,0).numpy())
                # plt.subplot(122)
                # plt.imshow(truth[0].permute(1,2,0).numpy())
                # plt.show()
                # exit(0)
                image.to(device)

                if verbose:
                    print(f'image shape is {image.shape}')
                    if torch.cuda.is_available():
                        image = image.cuda()

                # apply FRVSR
                hr_out, lr_out = model(image)
                hr_out = hr_out.clone()
                lr_out = image.clone()
                # plt.imshow(hr_out[0].permute(1,2,0).detach().numpy())
                # plt.imshow(lr_out[0].permute(1,2,0).detach().numpy())
                # plt.imshow(truth[0].permute(1,2,0).clone().numpy())
                #plt.show() #TODO uncomment
                print(image.shape)
                print(lr_out.shape)
                l1 = torch.mean((truth - hr_out) ** 2)
                l2 = torch.mean((image - lr_out) ** 2)
                print(l1)
                print(l2)
                # print(lr_out)
                # # print(image)
                # hr_out = Dataset_OnlyHR.inverse_transform(hr_out.clone())
                # lr_out = Dataset_OnlyHR.inverse_transform(lr_out.clone())
                # image = Dataset_OnlyHR.inverse_transform(image.clone())
                # truth = Dataset_OnlyHR.inverse_transform(truth.clone())
                hr_out = trunc(hr_out.clone())
                lr_out = trunc(lr_out.clone())
                aw_out = trunc(model.afterWarp.clone())

                out_psnr += psnr(hr_out, truth)
                l1 = torch.mean((truth - hr_out) ** 2)
                l2 = torch.mean((image - lr_out) ** 2)
                print(l1)
                print(l2)

                #plt.imshow(hr_out[0].permute(1, 2, 0).detach().numpy())
                #plt.imshow(truth[0].permute(1,2,0).clone().numpy())
                plt.imshow(lr_out[0].permute(1, 2, 0).detach().numpy())
                #plt.show()

                def output(out, writer):
                    out = out.clone()
                    out_img = out.data[0].numpy()
                    out_img *= 255.0
                    out_img = (np.uint8(out_img)).transpose((1, 2, 0))
                    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                    writer.write(out_img)

                # Write the high res video
                output(hr_out, hr_video_writer)

                # Write the low res video
                output(lr_out, lr_video_writer)

                # Write the intermediate warped video
                output(aw_out, aw_video_writer)

                # Write the ground truth video
                output(truth, gt_video_writer)

            hr_video_writer.release()
            lr_video_writer.release()
            aw_video_writer.release()
            gt_video_writer.release()
            print(f"PSNR is {out_psnr / 7}")
            tot_psnr = (tot_psnr * (idx - 1) + out_psnr / 7) / idx
            break
    print(f"Average PSNR is {tot_psnr}")
