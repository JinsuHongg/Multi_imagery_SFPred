import os
import glob
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# pytorch package
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# predefined class
from .Models import mobilenet, ResNet18, ResNet34, ResNet50
from .Training import SolarFlSets, HSS2, TSS, F1Pos, HSS_multiclass, TSS_multiclass, train_loop, test_loop_cp, oversample_func
from .Cp import compute_cov_length, conformity_score

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
print('1st check cuda..')
print('Number of available device', torch.cuda.device_count())
print('Current Device:', torch.cuda.current_device())
print('Device:', device)

# create parser here
parser = argparse.ArgumentParser(description="FullDiskModelTrainer")
parser.add_argument("--batch_size", type = int, default = 64, help = "batch size")
parser.add_argument("--freeze", type = bool, default = False, help = 'Enter True or False to freeze the convolutional layers')
parser.add_argument("--data", type = str, default = 'Het', help = "Enter Data source: EUV304, HMI-CTnuum, HMI-Mag, Het")
opt = parser.parse_args()

# define base directory here
img_dir = {"EUV-304" : "/workspace/data/hetero_data/euv/compressed/304", 
           "HMI-CTnuum" : "/workspace/data/hetero_data/hmi/compressed/continuum/",
           "HMI-Mag" : "/workspace/data/hetero_data/hmi/compressed/mag"}

# define data directory
if opt.data == "EUV304":
    channel_tag = 'EUV-304'
    img_dir = img_dir[channel_tag]
    print("Data Selected:", opt.data)
    
elif opt.data == "HMI-CTnuum":
    channel_tag = 'HMI-CTnuum'
    img_dir = img_dir[channel_tag]
    print("Data Selected:", opt.data)
   
elif opt.data == "HMI-Mag":
    channel_tag = 'HMI-Mag'
    img_dir = img_dir[channel_tag]
    print("Data Selected:", opt.data)
    
elif opt.data == 'Het':
    channel_tag = 'Het'
    print("Data Selected:", opt.data)

else:
    print("Data Selected: ", opt.data)
    print('Invalid data source')
    exit()

# hyper-parameter 
batch_size = opt.batch_size

# Define dataset here!
crr_path = os.getcwd()
save_path = crr_path + '/Multi_imagery_SFPred/Results/CV/'
file_path = crr_path + '/Multi_imagery_SFPred/Dataset/label/'

p = [1, 2, 3, 4]
# train_list = [f'24image_multi_GOES_classification_Partition{p[0]}.csv', 
#             f'24image_multi_GOES_classification_Partition{p[1]}.csv']
cp_file = f'24image_multi_GOES_classification_Partition{p[2]}.csv'
test_file = f'24image_multi_GOES_classification_Partition{p[3]}.csv'

# we don't need training set, we only need calibration and testing sets to create prediction sets.
# test set and calibration set
df_cal = pd.read_csv(file_path + cp_file)
df_test = pd.read_csv(file_path + test_file)

# string to datetime
df_cal['Timestamp'] = pd.to_datetime(df_cal['Timestamp'], format = '%Y-%m-%d %H:%M:%S')
df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'], format = '%Y-%m-%d %H:%M:%S')

# validation data loader
data_cal = SolarFlSets(annotations_df = df_cal, img_dir = img_dir, channel = channel_tag, normalization = True)
data_testing = SolarFlSets(annotations_df = df_test, img_dir = img_dir, channel = channel_tag, normalization = True)
cal_dataloader = DataLoader(data_cal, batch_size = batch_size, shuffle = True) # num_workers = 0, pin_memory = True, 
test_dataloader = DataLoader(data_testing, batch_size = batch_size, shuffle = False) # num_workers = 0, pin_memory = True,

# Settings
models = ['Mobilenet', 'Resnet18', 'Resnet34', 'Resnet50']

for model_name in models:
    # define model, loss, optimizer and scheduler
    # define model here
    if model_name == "Mobilenet":
        net = mobilenet().to(device)
        
    elif model_name == "Resnet18":
        net = ResNet18().to(device)
        
    elif model_name == "Resnet34":
        net = ResNet34().to(device)
        
    elif model_name == "Resnet50":
        net = ResNet50().to(device)

    else:
        print('Invalid Model')
        exit()

    # 1. First, let's check the keys in your state dict
    path = f'./Multi_imagery_SFPred/Results/CV/regmodel/{model_name}_{channel_tag}_freeze{opt.freeze}*.pth'
    file = glob.glob(path)[0]

    # becuase of the parallel processing we have different prix for the weights
    saved_state_dict = torch.load(file)['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k.replace("module.", "") # Remove the 'model.' prefix
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)

    print('--------------------------------------------------------------------------------')
    print(f'Model: {model_name}')
    print(f'file: {file}')
    print(f'Cp: {p[2]}, Test: {p[3]}')
    print('--------------------------------------------------------------------------------')
    print()

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    score_fn = nn.Softmax(dim=1)

    # Check Calibration set for conformity score
    print('Processing calibration..')
    t0 = time.time()
    cal_loss, cal_result = test_loop_cp(dataloader=cal_dataloader, model=net, loss_fn=loss_fn, score_fn=score_fn)
    duration = (time.time() - t0)/60
    print(f"Cal is done, {duration:.2f} min is spent")
    # table = confusion_matrix(cal_result[:, 1], cal_result[:, 0])
    # HSS_score = HSS_multiclass(table)
    # TSS_score = TSS_multiclass(table)
    # F1_score = f1_score(cal_result[:, 1], cal_result[:, 0], average='macro')

    # predict data
    print('Processing testset..')
    t0 = time.time()
    test_loss, test_result = test_loop_cp(dataloader=test_dataloader, model=net, loss_fn=loss_fn, score_fn=score_fn)
    duration = (time.time() - t0)/60
    print(f"Cal is done, {duration:.2f} min is spent")

    label = test_result[:, 4].astype('int')
    cp_result = []
    acp_result = []

    for i in range(0, 100): #confidence range from 0 to 99
        print(f'process confidence level {1*i}%')
        CP_arr, CP_arr_scores, quantile_cp = conformity_score(cal_result, test_result, quantile = 1-(0.01*i), mode = 'CP')
        ACP_arr, ACP_arr_scores, quantile_acp = conformity_score(cal_result, test_result, quantile = 1-(0.01*i), mode = 'AdpCP')

        cp_cov, cp_avg = compute_cov_length(label, CP_arr)
        acp_cov, acp_avg = compute_cov_length(label, ACP_arr)
        cp_empty = np.all(np.isin(CP_arr, 0), axis=1)
        acp_empty = np.all(np.isin(ACP_arr, 0), axis=1)
        num_cp_empty_set = len(cp_empty[cp_empty])
        num_acp_empty_set = len(acp_empty[acp_empty])

        cp_result.append([i, quantile_cp, cp_cov, cp_avg, num_cp_empty_set])
        acp_result.append([i, quantile_acp, acp_cov, acp_avg, num_acp_empty_set])

    cp_result = np.array(cp_result)
    acp_result = np.array(acp_result)

    savepath = crr_path + '/Multi_imagery_SFPred/Results/cp' 
    with open(savepath + f'/{model_name}_{channel_tag}_freeze{opt.freeze}_cp{p[2]}_test{p[3]}.npy', 'wb') as f:
            cal_sfmax_arr = np.save(f, cal_result)
            test_sfmax_arr = np.save(f, test_result)
            cp_result_arr = np.save(f, cp_result)
            acp_result_arr = np.save(f, acp_result)
