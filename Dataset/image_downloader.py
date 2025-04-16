import glob
import requests
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def download_from_helioviewer(
        basedir:str = '/data/hmi_compressd', 
        start:str = '2010-12-05 00:00:00', 
        stop:str = "2024-12-31 12:59:59", 
        cadence:int = 360, 
        source:int = 13, 
        search_space:int = 10,
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

    #File counter
    counter = 0

    #Start Date
    dt = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = datetime.datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")
    delta = datetime.timedelta(minutes = cadence)

    lis = []
    while dt <= end:
        
        final_date = str(dt.date()) + 'T' + str(dt.time()) + 'Z'
        Path(f'{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}').mkdir(parents=True, exist_ok=True)
        #Defining name of downloaded images based on the date and time
        filename = f"{file_prefix}.{dt.year}.{dt:%m.%d_%H.%M.%S}.jp2"
        file_loc_with_name = f'{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}/' + filename

        # check if the file already exists
        if os.path.exists(file_loc_with_name):
            print(filename, "File already exists. Passed!")
            dt = dt+ delta
            continue

        base_url = "https://api.helioviewer.org/v2/getJP2Image/"
        params = {
            "date": final_date,
            "sourceId": source,
            "jpip": True
        }

        # First request: Get JP2 URI (used to extract timestamp)
        response = requests.get(base_url, params=params)
        url = response.content.decode()  # decode instead of str() for clarity
        url_temp = url.rsplit('/', 1)[-1]
        date_received_str = url_temp.rsplit('__', 1)[0][:-4]
        received = datetime.datetime.strptime(date_received_str, "%Y_%m_%d__%H_%M_%S")

        print(f"Requested date: {dt}", f"Response data: {received}")
        Path(f'{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}/').mkdir(parents=True, exist_ok=True)

        # Check if the received image is within the desired time window
        if abs(received - dt) <= datetime.timedelta(minutes=search_space):
            # Second request: Actually download the image (no jpip)
            img_response = requests.get(base_url, params={"date": final_date, "sourceId": source})
            with open(file_loc_with_name, 'wb') as f:
                f.write(img_response.content)

            lis.append([dt, received])
            counter += 1
            if counter % 2500 == 0:
                print(f"{counter} Files Downloaded")

        # go to next timestamp
        dt = dt+ delta

    #Total Files Downloaded
    print('Total Files Downloaded: ', counter)

def image_resize(filepath:str, width, height):
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return image

def jp2_to_jpg_conversion(source = '/data/hmi_compressd/', destination = '/data/hmi_jpgs/', file_prefix = "HMI-mig", resize=False, width=512, height=512):
    """
    This function reads the jp2s stored inside the source directory into jpgs and store in destination directory

    1) source: the source directory which the HMI magnetograms are stored in.
    2) destination: the destination directory which the compressed HMI magnetograms exist
    3) resize: if resize =  True, it will resize the jpgs to specified dimension.
    4) width: image resize width
    5) height: image resize height
    """

    files = glob.glob(source + f"*/*/*/*.jp2")
    files = sorted(files)
    for i, file in enumerate(files):

        year, month, day, f_n = file.split("/")[6:]
        file_name = f_n[:-3] + "jpg"
        full_path = os.path.join(destination, year, month, day, file_name)

        if os.path.exists(full_path):
            print(file, "already exists. Passed!")
            continue

        try:
            Path(f'{destination}/{year}/{month}/{day}/').mkdir(parents=True, exist_ok=True)
            image = image_resize(file, width=width, height=height)
            cv2.imwrite(full_path, image)
            print(i, full_path, 'Converted')
        except:
            print(i, full_path, 'Error Occured!')
            pass

def convert_jp2_to_jpg(file:str, destination:str, width:int, height:int):
    try:
        parts = Path(file).parts
        year, month, day, f_n = parts[-4:]
        file_name = f_n[:-3] + "jpg"
        full_path = Path(destination) / year / month / day / file_name

        if full_path.exists():
            print(file, "already exists. Passed!")
            return

        full_path.parent.mkdir(parents=True, exist_ok=True)
        image = image_resize(file, width=width, height=height)
        cv2.imwrite(str(full_path), image)
        print(full_path, 'Converted')

    except Exception as e:
        print(file, 'Error Occurred!', e)

if __name__ == '__main__':

    map_source_id = {
        "94": 8, 
        "131":9, 
        "171":10, 
        "193":11, 
        "211":12, 
        "304": 13, 
        "335":14, 
        "hmi": 19
        }

    for channel in ["94", "131", "171", "193", "211", "335"]:

        prefix = f'AIA{channel}'
        save_dir = f'/workspace/data/hetero_data/euv/{channel}'
        
        download_from_helioviewer(
            basedir = save_dir,
            start = "2010-12-05 00:00:00",
            stop = "2024-12-31 12:59:59",
            cadence = 60, 
            source = map_source_id[channel], 
            file_prefix = prefix
            )
            
        files = glob.glob(save_dir + f"/*/*/*/*.jp2")
        files = sorted(files)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(convert_jp2_to_jpg, file) for file in files]

            for _ in tqdm(as_completed(futures), total=len(files), desc="Converting"):
                pass


        # jp2_to_jpg_conversion(
        #     source = save_dir + '/',
        #     file_prefix = prefix, 
        #     destination = f'/workspace/data/hetero_data/euv/compressed/{channel}/', 
        #     resize = True, 
        #     width = 512, 
        #     height = 512
        #     )
