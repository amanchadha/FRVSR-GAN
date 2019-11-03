cd ~/Documents/iSeeBetter/FR-SRGAN 
python3 checkTrain.py

cd ~/Documents/iSeeBetter/FR-SRGAN
python3 Test_iSeeBetter.py --video out_srf_original_random_sample.mp4

cd ~/Documents/iSeeBetter/FR-SRGAN/SRGAN
python3 test_video.py --video ../out_srf_4_out_srf_original_random_sample.mp4