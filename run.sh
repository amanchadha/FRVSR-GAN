# iSeeBetter: A Novel Approach to Video Super-Resolution using Adaptive Frame Recurrence and Generative Adversarial Networks
# aman@amanchadha.com

# generate a low res random sample and apply FRVSR
python3 checkTrain.py

# test
python3 Test_iSeeBetter.py --video FRSRVOut_LowRes_Random_Sample.mp4

# apply SRGAN
cd SRGAN
python3 test_video.py --video ../FRSRVOut_LowRes_Random_Sample.mp4