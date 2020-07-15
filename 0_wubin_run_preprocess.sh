#! /bin/bash


##### setup train and test directories

mkdir input
mv *.rar input
cd input
unrar e *.rar
mv *.rar ../
mkdir train test
mv test_* test
mv *.wav train
cd train
mkdir hug hungry uncomfortable sleepy diaper awake
mv hug_* hug
mv hungry_* hungry
mv uncomfortable_* uncomfortable
mv sleepy_* sleepy
mv diaper_* diaper
mv awake_* awake

##### set up validation data to input/val

cd ..
mkdir val
mkdir val/hug val/hungry val/uncomfortable val/sleepy val/diaper val/awake
mkdir eda
# 1600 means "1600Hz sampling rate"
mkdir eda/1600

mkdir eda/1600/hug eda/1600/hungry eda/1600/uncomfortable eda/1600/sleepy eda/1600/diaper eda/1600/awake

cd ..

# MOVE 1600 Hz sampling rate .wav files to corresponding folder

python3 move_1600.py

# seperate 10 (.wav files, randomly chosen) * 6 (classes) into folder input/val for validation

python3 move_val.py


