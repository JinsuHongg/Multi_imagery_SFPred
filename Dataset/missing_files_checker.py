
import glob
import datetime
import pandas as pd


start_date = '2010-12-05 00:00:00'
stop_date = datetime.datetime.now()
dt = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

total_result = []

while dt <= stop_date:

    time_text = f"{dt.year}.{dt.month:02d}.{dt.day:02d}_{dt.hour:02d}.{dt.minute:02d}.{dt.second:02d}"
    base = [time_text, 0, 0, 0]

    file_exist = f"/workspace/data/hetero_data/**/**/{dt.year}/{dt.month:02d}/{dt.day:02d}/*." + \
                    time_text + '.jp2' 
    for file in glob.glob(file_exist):

        if 'EUV-304' in file:
            base[1] = 1

        elif 'HMI-Mag' in file:
            base[2] = 1
        
        elif 'HMI-CTnuum' in file:
            base[3] = 1
        
    total_result.append(base)
    print(base)
    
    dt += datetime.timedelta(minutes = 60)

df = pd.DataFrame(total_result, columns = ['Timestamp', 'EUV304', 'HMI_Mag', 'HMI_CTnuum'])
df.to_csv("Missing_info.csv", index = False)

# while dt < stop_date:
        
#         # Define Timestamp
#         filename = str(dt.year) + '.' +  f'{dt.month:02d}' + '.' + f'{dt.day:02d}' + '_'\
#             + f'{dt.hour:02d}' + '.' + f'{dt.minute:02d}' + '.' + f'{dt.second:02d}'
        
#         file_exit = "/worksapce/data/hetero/**/**/{dt.year}/{dt.month:02d}/{dt.day:02d}/*." + \
#                     "{dt.year}.{dt.month:02d}.{dt.day:02d}_{dt.hour:02d}.{dt.minute:02d}.{dt:second:02d}.jp2"

        
#         # Path(f'{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}').mkdir(parents=True, exist_ok=True)
#         #Defining name of downloaded images based on the date and time
        


#         hours = datetime.timedelta(minutes = cadence)
#         dt = dt+ hours