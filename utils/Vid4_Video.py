"""
This file contains implementation of dataset classes.
Aman Chadha | aman@amanchadha.com

Adapted from FR-SRGAN, MIT 6.819 Advances in Computer Vision, Nov 2018
"""

import cv2
import os

lr_img_root = "Data/FRVSR_VID4/LR/walk"
hr_img_root = "Data/FRVSR_VID4/HR/walk"

# Edit each frame's appearing time!
fps = 24
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4','v')
videoWriter_hr = cv2.VideoWriter("Data/hr_test.mp4", fourcc, fps, (704, 480))
videoWriter_lr = cv2.VideoWriter("Data/lr_test.mp4", fourcc, fps, (176, 120))


lr_im_names = os.listdir(lr_img_root)
lr_im_names.sort()

#print(lr_im_names)

for im_name in lr_im_names:
    # print(im_name)
    # print(os.path.join(lr_img_root, str(im_name)))
    if cv2.imread(os.path.join(lr_img_root,im_name)) is not None:
        # print(os.path.join(lr_img_root,str(im_name)))
        frame = cv2.imread(os.path.join(lr_img_root,str(im_name)))
        # print(frame.shape)
        cv2.imshow('frame', frame)
        #frame = (np.uint8(frame)).transpose((1, 2, 0))
        videoWriter_lr.write(frame)
    else:
        pass

hr_im_names = os.listdir(hr_img_root)
hr_im_names.sort()
for im_name in hr_im_names:
    if cv2.imread(os.path.join(hr_img_root,im_name)) is not None:
        frame = cv2.imread(os.path.join(hr_img_root,str(im_name)))
        #frame = (np.uint8(frame)).transpose((1, 2, 0))
        #frame = cv2.resize(frame, (704, 480))
        videoWriter_hr.write(frame)
    else:
        pass


videoWriter_lr.release()
videoWriter_hr.release()
cv2.destroyAllWindows()
