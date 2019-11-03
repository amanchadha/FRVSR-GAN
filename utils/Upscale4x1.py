"""
This file creates a 4x1 upscaled video.
aman@amanchadha.com

Credits to FR-SRGAN (https://github.com/ReNginx/FR-SRGAN)
"""

from checkTrain import psnr
import cv2
from torchvision.transforms import ToTensor, ToPILImage

hr = cv2.VideoCapture('out_srf_original_random_sample.mp4')
gt = cv2.VideoCapture('out_srf_groundtruth_1_random_sample.mp4')

out_psnr = 0
cnt = 0

while hr.isOpened() and gt.isOpened():
    hrRet, hrframe = hr.read()
    gtRet, gtframe = gt.read()
    if hrRet and gtRet:
        hrframe = cv2.resize(hrframe, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        hrframe = ToTensor()(hrframe).unsqueeze(0)
        gtframe = ToTensor()(gtframe).unsqueeze(0)
        cnt += 1
        out_psnr += psnr(hrframe, gtframe)
    else:
        break
print(f'PSNR: {out_psnr / cnt}')
hr.release()
gt.release()
