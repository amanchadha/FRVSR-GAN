import cv2
#from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
import numpy as np
import scipy.misc


img_root = "Data/FRVSR_VID4/FRVSR/calendar"
w_point = 150 # set the weight-coordinate for generating the "line"
img_height = 576

im_names = os.listdir(img_root)
im_names.sort()


p_length = len(im_names)
print(p_length)

profile = np.zeros(shape = (p_length,img_height,3))
#print(profile.shape)
i = 0
for im_name in im_names:
    # print(im_name)
    # print(os.path.join(lr_img_root, str(im_name)))
    if cv2.imread(os.path.join(img_root,im_name)) is not None:
        # print(os.path.join(lr_img_root,str(im_name)))
        frame = cv2.imread(os.path.join(img_root,str(im_name)))
        line = frame[:,w_point,:]
        #print(line.shape)
        #print(profile[i].shape)
        profile[i,:,:] = line
        i = i + 1
        #print(line.shape)
        #print(frame.shape)
        #cv2.imshow('frame', line)
        #cv2.waitKey(0)
        #frame = (np.uint8(frame)).transpose((1, 2, 0))
        #videoWriter_lr.write(frame)
    else:
        pass

print(profile)
print(profile.shape)

cv2.imwrite('profile2.jpg', profile)
cv2.imshow("profile", profile);
#cv2.waitKey();
