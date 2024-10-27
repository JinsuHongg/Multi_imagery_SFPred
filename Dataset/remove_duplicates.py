import os
import glob

dir = '/workspace/data/hetero_data/hmi/mag/**/**/**/*'
for file in glob.glob(dir):
    if "HMI-mag.m" in file:
        name = file.split('/')[-1]
        print(f'{name} is deleted!')
        os.remove(file)