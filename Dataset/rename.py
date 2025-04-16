import os
import glob


path = '/workspace/data/hmi_jpgs_512/'

for year in range(2011, 2015):
    for month in range(1, 13):
        for day in range(1, 32):
            files = glob.glob(path + f'{year}/{month:02d}/{day:02d}/*HMI.m.*.jpg')
            # print(path + '{year}/{month:02d}/{day:02d}/*HMI.m.*.jpg')
            # if len(files) >= 1:
            #     print(files)

            for file in files:
                f_path, timetag = file.split('HMI.m.')
                print(file, end =' || ')
                print(f_path + 'HMI.m' + timetag)
                os.rename(file, f_path + 'HMI.m' + timetag)