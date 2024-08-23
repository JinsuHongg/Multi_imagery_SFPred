import os
import glob

dir = '/workspace/data/hetero_data/euv/304/**/**/**/*'
for file in glob.glob(dir):
    if "AIA304.m" in file:
        name = file.split('/')[-1]
        print(f'{name} is deleted!')
        os.remove(file)