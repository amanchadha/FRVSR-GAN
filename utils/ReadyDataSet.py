"""
This file structures the dataset so it can be consumed by the algorithm.

- Download dataset from: http://data.csail.mit.edu/tofu/testset/vimeo_super_resolution_test.zip
- Run this code for LR and HR seperately to form a sorted data folder for convenience
- Useful: how to detract the .DS_Store
    # https://macpaw.com/how-to/remove-ds-store-files-on-mac
    # find . -name '.DS_Store' -type f -delete

Aman Chadha | aman@amanchadha.com

Adapted from FR-SRGAN, MIT 6.819 Advances in Computer Vision, Nov 2018
"""

import os, sys, shutil

source_folder = "/Users/Aman/Documents/iSeeBetter/FR-SRGAN/Data/vimeo_test_clean/sequences/"

if os.path.exists(source_folder + ".DS_Store"):
    os.remove(source_folder + ".DS_Store")
else:
    print(".DS_Store does not exist")

video_list = os.listdir(source_folder)
video_list.sort()

counter = 0

for video in video_list:
    video_path = os.path.join(source_folder, video)
    frames_list = os.listdir(video_path)
    frames_list.sort()
    for frames in frames_list:
        frames_path = os.path.join(video_path,frames)
        counter = counter + 1
        new_frames_name = counter
        des = "Data/HR/" + str(new_frames_name)
        #des = "Data/LR/" + str(new_frames_name)
        print(des)
        shutil.copy(frames_path, des)