"""
This file fetches and structures the dataset so it can be consumed by the algorithm.

- Download Vimeo90K dataset (get the original test set - not downsampled or downgraded by noise) from:
  http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip
- Run this code for LR and HR seperately to form a sorted data folder for convenience
- To delete all the .DS_Store files: find . -name '.DS_Store' -type f -delete

aman@amanchadha.com
"""

import os, sys, shutil, urllib.request
from tqdm import tqdm
from zipfile import ZipFile

DATASET_URL = "http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip"
DATA_FOLDER = "Data"
SOURCE_PATH = os.path.join(DATA_FOLDER, "vimeo_test_clean")
DEST_PATH = os.path.join(DATA_FOLDER, "HR")

# Create a data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    try:
        os.mkdir(DATA_FOLDER)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

class downloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def downloadURL(url, output_path):
    with downloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# If the dataset doesn't exist, download and extract it
if not os.path.exists(SOURCE_PATH):
    # Fetch the dataset if it hasn't been downloaded yet
    if not os.path.exists(SOURCE_PATH + '.zip'):
        downloadURL(DATASET_URL, os.path.join(DATA_FOLDER, "vimeo_test_clean.zip"))

    # Extract it
    print(os.path.join(DATA_FOLDER, 'vimeo_test_clean.zip'))

    with ZipFile(os.path.join(DATA_FOLDER, 'vimeo_test_clean.zip'), 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        try:
            zipObj.extractall(DATA_FOLDER)
        except BadZipFile:
            # Re-download the file
            downloadURL(DATASET_URL, os.path.join(DATA_FOLDER, "vimeo_test_clean.zip"))
            zipObj.extractall(DATA_FOLDER)
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