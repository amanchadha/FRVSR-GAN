"""
This file fetches and structures the dataset so it can be consumed by the algorithm.

- Download Vimeo90K dataset (get the original test set - not downsampled or downgraded by noise) from:
  http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip
- Run this code for LR and HR seperately to form a sorted data folder for convenience
- To delete all the .DS_Store files: find . -name '.DS_Store' -type f -delete

aman@amanchadha.com
"""

import os, sys, shutil, urllib.request
from zipfile import ZipFile

DATASET_URL = "http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip"
SOURCE_PATH = "Data/vimeo_test_clean"
DEST_PATH = "Data/HR"

# If the dataset doesn't exist, download and extract it
if not os.path.exists(SOURCE_PATH + '.zip'):
    pass
    # Fetch the dataset
    urllib.request.urlretrieve(DATASET_URL, "Data/vimeo_test_clean.zip")

    # Extract it
    ZipFile.extractall(path="Data/vimeo_test_clean.zip")
else:
    # Recursively remove all the ".DS_Store files"
    for currentPath, _, currentFiles in os.walk(SOURCE_PATH):
        if ".DS_Store" in currentFiles:
            os.remove(os.path.join(currentPath, ".DS_Store"))

    # Make a list of video sequences
    sequencesPath = os.path.join(SOURCE_PATH, "sequences")
    videoList = os.listdir(sequencesPath)
    videoList.sort()

    count = 0
    for video in videoList:
       videoPath = os.path.join(sequencesPath, video)
       framesList = os.listdir(videoPath)
       framesList.sort()

       for frames in framesList:
           frames_path = os.path.join(videoPath, frames)
           count += 1
           new_frames_name = count
           des = os.path.join(DEST_PATH, str(new_frames_name))
           print("Creating: ", des)
           shutil.copytree(frames_path, des)