import requests
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os, csv
import cv2

def download_from_helioviewer(basedir:str = '/data/hmi_compressd', cadence:int = 360, source:int = 13, file_prefix:str = 'AIA304'):
    """
    1)
    This functions download 4k magnetogram jp2s from helioviewer api (https://api.helioviewer.org/v2/getJP2Image/) at a cadence of 12mins as available.
    Inside the basedir/ : downloaded magnetograms are stored creating a heirarchy as:
    basedir/
        year/
            month/
                day/filename.jp2

    The filename are renamed as: HMI.{year}.{month}.{day}_{hour}.{minute}.{second}.jp2
    2) stop_year: the downloading stops until current time.
    3) cadence: Images are downloaded with interval (or cadence). Default: 60min. Change it with your purposes. 

    """

    start_date = '2010-12-05 00:00:00'
    stop_date = datetime.datetime.now()
    #File counter
    counter = 0

    #Start Date
    dt = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    lis = []
    while True:
        hours = datetime.timedelta(minutes = cadence)
        dt = dt+ hours
        final_date = str(dt.date()) + 'T' + str(dt.time()) + 'Z'
        if dt >= stop_date:
            break
        Path(f'{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}').mkdir(parents=True, exist_ok=True)
        #Defining name of downloaded images based on the date and time
        filename = file_prefix + '.' + str(dt.year) + '.' +  f'{dt.month:02d}' + '.' + f'{dt.day:02d}' + '_'\
            + f'{dt.hour:02d}' + '.' + f'{dt.minute:02d}' + '.' + f'{dt.second:02d}' + '.jp2'
        file_loc_with_name = f'{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}/' + filename

        #Using jpip=True gives the uri of the image which we use to parse time_stamp of available image
        #Detail documentation is provided on api.helioviewer.org
        # requestString = "https://api.helioviewer.org/ss/getJP2Image/?date=" + final_date + f"&sourceId={source}&jpip=true"
        requestString = "https://api.helioviewer.org/v2/getJP2Image/?date=" + final_date + f"&sourceId={source}&jpip=true"
        #print(requestString, '-->Requested')

        #Parsing date from the recived uri
        response = requests.get(requestString)
        url = str(response.content)
        url_temp = url.rsplit('/', 1)[-1]
        date_recieved = url_temp.rsplit('__', 1)[0][:-4]
        recieved = datetime.datetime.strptime(date_recieved, "%Y_%m_%d__%H_%M_%S")
        print(f'Requested date: {dt}', f'Response data: {recieved}')
        
        #Now comparing the timestamp of available image and requested image
        #Download only if within the window of 12 minutes.
        if abs(recieved-dt) <= (datetime.timedelta(minutes = 10)):
            
            #This uri provides access to the actual resource ( i.e., images)
            request_uri = "https://api.helioviewer.org/v2/getJP2Image/?date=" + final_date + f"&sourceId={source}"
            hmidata = requests.get(request_uri)
            open(file_loc_with_name,'wb').write(hmidata.content)
    #         print(final_date, '-->Downloaded')
            lis.append([dt, recieved])
            counter += 1
            if counter % 2500 == 0:
                print(counter, 'Files Downloaded')

    #Total Files Downloaded
    print('Total Files Downloaded: ', counter)

def jp2_to_jpg_conversion(source = '/data/hmi_compressd/', destination = '/data/hmi_jpgs/', file_prefix = "HMI-mig", resize=False, width=512, height=512):
    """
    This function reads the jp2s stored inside the source directory into jpgs and store in destination directory

    1) source: the source directory which the HMI magnetograms are stored in.
    2) destination: the destination directory which the compressed HMI magnetograms exist
    3) resize: if resize =  True, it will resize the jpgs to specified dimension.
    4) width: image resize width
    5) height: image resize height
    """
    filename = f'totalfiles_{file_prefix}.csv'
    with open(filename,'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        for path, subdirs, files in os.walk(source):
            for name in files:
                w.writerow([os.path.join(path, name), os.path.getsize(os.path.join(path, name))])
    colnames=['path', 'size'] 
    df = pd.read_csv(filename, header=None, names=colnames)
    df['timestamp'] = df['path'].str.rsplit("/", n=2, expand=True).iloc[:, -1].str[len(file_prefix) + 1 : -4]
    lis=  []
    for i in range(len(df)):
        dt = datetime.datetime.strptime(df['timestamp'][i], '%Y.%m.%d_%H.%M.%S')
        Path(f'{destination}/{dt.year}/{dt.month:02d}/{dt.day:02d}').mkdir(parents=True, exist_ok=True)
        #Defining name of downloaded images based on the date and time
        filename = file_prefix + '.' + str(dt.year) + '.' +  f'{dt.month:02d}' + '.' + f'{dt.day:02d}' + '_'\
            + f'{dt.hour:02d}' + '.' + f'{dt.minute:02d}' + '.' + f'{dt.second:02d}' + '.jpg'
        file_loc_with_name = f'{destination}/{dt.year}/{dt.month:02d}/{dt.day:02d}/' + filename
        try:
            image = cv2.imread(str(df.path[i]), cv2.IMREAD_UNCHANGED)
            if resize:
                image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
            cv2.imwrite(file_loc_with_name, image)
            print(i, df.timestamp[i], 'Converted')
        except:
            print('Error Occured!')
            lis.append([df['path'][i]])
            pass

if __name__ == '__main__':
    prefix = 'EUV-304'
    save_dir = '/workspace/data/hetero_data/euv/304'
    # download_from_helioviewer(basedir = save_dir, cadence = 60, source = 13, file_prefix = prefix)
    jp2_to_jpg_conversion(source = save_dir + '/',
                          file_prefix = prefix, 
                          destination = '/workspace/data/hetero_data/euv/compressed/304/', 
                          resize = True, width = 512, height = 512)
