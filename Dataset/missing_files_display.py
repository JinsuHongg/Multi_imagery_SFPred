import pandas as pd

df = pd.read_csv('Missing_info.csv')
print('EUV304 missing:', len(df.query('EUV304 == 0')))
print('HMI Magnetograms missing:', len(df.query('HMI_Mag == 0')))
print('HMI Continuum missing:', len(df.query('HMI_CTnuum == 0')))
print('HMI Continuum missing:', len(df.query('HMI_CTnuum == 1 & HMI_Mag == 1 & EUV304 == 1')))