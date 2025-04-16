import glob
import requests
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os, csv
import cv2

def download_from_helioviewer(
    start_date:str = '2014-01-01 00:00:00',
    stop_date:str = '2014-12-31 12:59:59',
    basedir:str = '/data/hmi_compressd', 
    cadence:int = 360,
    time_margin:int = 10, 
    source:int = 13, 
    file_prefix:str = 'AIA304'
    ):

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

    #Start/stop Date
    dt = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    stop_date = datetime.datetime.strptime(stop_date, "%Y-%m-%d %H:%M:%S")#datetime.datetime.now()
    delta = datetime.timedelta(minutes = cadence)

    counter = 0
    downloaded_files = []

    while dt <= stop_date:
        # Create directory structure
        dir_path = Path(f"{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}")
        dir_path.mkdir(parents=True, exist_ok=True)

        # Define filename
        filename = f"{file_prefix}{dt.year}.{dt.month:02d}.{dt.day:02d}_{dt.hour:02d}.{dt.minute:02d}.{dt.second:02d}.jp2"
        file_path = dir_path / filename

        if file_path.exists():
            print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} already exists, skipping...")
            counter += 1
            dt += delta
            continue

        # Construct API request
        final_date = f"{dt.date()}T{dt.time()}Z"
        request_url = f"https://api.helioviewer.org/v2/getJP2Image/?date={final_date}&sourceId={source}&jpip=true"

        try:
            response = requests.get(request_url)
            response.raise_for_status()
            url_temp = response.content.decode().rsplit('/', 1)[-1]
            date_received_str = url_temp.rsplit('__', 1)[0][:-4]
            received_dt = datetime.datetime.strptime(date_received_str, "%Y_%m_%d__%H_%M_%S")
            time_diff = abs(received_dt - dt)

            print(f"Requested: {dt}, Received: {received_dt}, Difference: {time_diff}")

            # Download only if the time difference is within 10 minutes
            if time_diff <= datetime.timedelta(minutes=time_margin):
                img_request_url = f"https://api.helioviewer.org/v2/getJP2Image/?date={final_date}&sourceId={source}"
                img_data = requests.get(img_request_url)
                img_data.raise_for_status()

                with open(file_path, 'wb') as f:
                    f.write(img_data.content)

                downloaded_files.append([dt, received_dt])
                counter += 1

                if counter % 2500 == 0:
                    print(f"{counter} files downloaded.")

        except (requests.RequestException, ValueError) as e:
            print(f"Error processing {dt}: {e}")

        dt += delta

    print(f"Total Files Downloaded: {counter}")
    return downloaded_files

def jp2_to_jpg_conversion(
    start_date:str = '2013-01-01 00:00:00',
    stop_date:str = '2013-12-31 12:59:59',
    cadence:int = 12,
    source:str = '/data/hmi_compressd/', 
    destination:str = '/data/hmi_jpgs/', 
    file_prefix:str = "HMI-mig", 
    resize:bool = False, 
    width:int = 512, 
    height:int = 512
    ):

    """
    This function reads the jp2s stored inside the source directory into jpgs and store in destination directory

    1) source: the source directory which the HMI magnetograms are stored in.
    2) destination: the destination directory which the compressed HMI magnetograms exist
    3) resize: if resize =  True, it will resize the jpgs to specified dimension.
    4) width: image resize width
    5) height: image resize height
    """
    
    # Processing files with proper cadence
    # this prevents from unnecessary files
    cadence_list = []
    time = datetime.datetime.strptime("00:00", "%H:%M")
    stop_time = datetime.datetime.strptime("23:59", "%H:%M")
    j=1
    while time < stop_time:
        cadence_list.append(f"{time.hour:02d}.{time.minute:02d}")
        time = time + datetime.timedelta(minutes = cadence) 

    #Start/stop Date
    dt = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    stop_dt = datetime.datetime.strptime(stop_date, "%Y-%m-%d %H:%M:%S")
    i = 0
    for year in range(dt.year, stop_dt.year + 1):
        for month in range(1, 13):
            for day in range(1, 32):
                files = glob.glob(source + f"{year}/{month:02d}/{day:02d}/*.jp2")
                if len(files) >= 1:
                    Path(destination + f'/{year}/{month:02d}/{day:02d}').mkdir(parents=True, exist_ok=True)
                
                for file in files:
                    f_path, timetag = file.split(file_prefix)
                    if timetag.split('_')[1][0:5] not in cadence_list:
                        continue

                    file_name = file_prefix + timetag[:-3] + "jpg"
                    file_loc_with_name = destination + f'/{year}/{month:02d}/{day:02d}/' + file_name
                    # print(file_loc_with_name)

                    # if file already exists, then pass this file
                    if os.path.exists(file_loc_with_name):
                        continue

                    try:
                        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                        if resize:
                            image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
                        cv2.imwrite(file_loc_with_name, image)
                        print(i, file, 'Converted')
                    except:
                        print(i, file, 'Error Occured!')
                        pass
                    i += 1

if __name__ == '__main__':
    prefix = 'HMI.m'
    start_date = '2014-05-06 00:00:00'
    stop_date = '2014-05-07 12:59:59'
    source_dir = '/workspace/data/hmi_jpgs_512/jp2000'
    cadence = 60
    download_from_helioviewer(
        start_date = start_date,
        stop_date = stop_date,
        basedir = source_dir,
        time_margin = 10, 
        cadence = cadence, 
        source = 19, 
        file_prefix = prefix
        )
    jp2_to_jpg_conversion(
        start_date = start_date,
        stop_date = stop_date,
        cadence = cadence,
        source = source_dir + '/',
        file_prefix = prefix, 
        destination = '/workspace/data/hmi_jpgs_512', 
        resize = True, 
        width = 512, 
        height = 512
        )
