# basic package
import math
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# pytorch package
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
#from pytorch_forecasting.metrics import QuantileLoss

# predefined class
from .Models import mobilenet, ResNet18, ResNet34, ResNet50
from .Training import SolarFlSets, HSS2, TSS, F1Pos, HSS_multiclass, TSS_multiclass, train_loop, test_loop, oversample_func

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
print('1st check cuda..')
print('Number of available device', torch.cuda.device_count())
print('Current Device:', torch.cuda.current_device())
print('Device:', device)

# dataset partitions and create data frame
print('2nd process, loading data...')

# create parser here
parser = argparse.ArgumentParser(description="FullDiskModelTrainer")
# parser.add_argument("--fold", type = int, default = 1, help = "Fold Selection")
parser.add_argument("--epochs", type = int, default = 4, help = "number of epochs")
parser.add_argument("--batch_size", type = int, default = 64, help = "batch size")
parser.add_argument("--lr", type = float, default = 1e-6, help = "learning rate")
parser.add_argument("--weight_decay", type = list, default = [0, 1e-4], help = "regularization parameter")
parser.add_argument("--max_lr", type = float, default = 1e-2, help = "MAX learning rate")
parser.add_argument("--models", type = str, default = 'Mobilenet', help = "Enter Mobilenet, Resnet18, Resnet34, Resnet50")
parser.add_argument("--freeze", type = bool, default = False, help = 'Enter True or False to freeze the convolutional layers')
parser.add_argument("--data", type = str, default = 'EUV304', help = "Enter Data source: EUV304, HMI-CTnuum, HMI-Mag, All")
opt = parser.parse_args()


# define dataset here
img_dir = {"EUV-304" : "/workspace/data/hetero_data/euv/compressed/304", 
           "HMI-CTnuum" : "/workspace/data/hetero_data/hmi/compressed/continuum/",
           "HMI-Mag" : "/workspace/data/hetero_data/hmi/compressed/mag"}

if opt.data == "EUV304":
    channel_tag = 'EUV-304'
    img_dir = img_dir[channel_tag]
    print("Data Selected: ", opt.data)
    
elif opt.data == "HMI-CTnuum":
    channel_tag = 'HMI-CTnuum'
    img_dir = img_dir[channel_tag]
    print("Data Selected: ", opt.data)
   
elif opt.data == "HMI-Mag":
    channel_tag = 'HMI-Mag'
    img_dir = img_dir[channel_tag]
    print("Data Selected: ", opt.data)
    
elif opt.data == 'All':
    # img_dir = img_dir["Magnetogram"]
    channel_tag = 'Het'
    print("Data Selected: ", opt.data)

else:
    print("Data Selected: ", opt.data)
    print('Invalid data source')
    exit()


save_path = '/workspace/Project/Multi_imagery_SFPred/Results/CV/'
file_path = '/workspace/Project/Multi_imagery_SFPred/Dataset/label/'

# Settings
model_name = opt.models

# hyper-parameter 
batch_size = opt.batch_size
num_epoch = opt.epochs
lr = opt.lr
max_lr = opt.max_lr
decay_val = opt.weight_decay

print(f'Hyper parameters: batch_size: {batch_size}, number of epoch: {num_epoch},' + \
      f'learning rate: {lr}, max learning rate: {max_lr}, decay value: {decay_val}')


# Cross-validatation with optimization ( total = 4folds X Learning rate sets X weight decay sets )
for wt in decay_val:
    training_result = []
    for i in range(0, 4):
        
        '''
        [ Grid search start here ] 
        - Be careful with  result array, model, loss, and optimizer
        - Their position matters

        '''

        p = [1, 2, 3, 4]
        test_p = p.pop(i)

        # Define dataset here! 
        train_list = [f'24image_multi_GOES_classification_Partition{p[0]}.csv', 
                    f'24image_multi_GOES_classification_Partition{p[1]}.csv', 
                    f'24image_multi_GOES_classification_Partition{p[2]}.csv']
        
        test_file = f'24image_multi_GOES_classification_Partition{test_p}.csv'
        
                
        print('--------------------------------------------------------------------------------')
        print(f'Train: ({p[0]}, {p[1]}, {p[2]}), Test: {test_p}')
        print(f"learning rate: {lr:.1e}, decay value: {wt:.1e}")
        print('--------------------------------------------------------------------------------')

        # train set
        df_train = pd.DataFrame([], columns = ['Timestamp', 'GOES_cls', 'Label'])
        for partition in train_list:
            d = pd.read_csv(file_path + partition)
            df_train = pd.concat([df_train, d])

        # test set and calibration set
        df_test = pd.read_csv(file_path + test_file)

        # string to datetime
        df_train['Timestamp'] = pd.to_datetime(df_train['Timestamp'], format = '%Y-%m-%d %H:%M:%S')
        df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'], format = '%Y-%m-%d %H:%M:%S')

        # training data loader
        # over/under sampling
        data_training, imbalance_ratio = oversample_func(df = df_train, img_dir = img_dir, channel = channel_tag, norm = True)

        # validation data loader
        data_testing = SolarFlSets(annotations_df = df_test, img_dir = img_dir, channel = channel_tag, normalization = True)
        train_dataloader = DataLoader(data_training, batch_size = batch_size, shuffle = True) # num_workers = 0, pin_memory = True, 
        test_dataloader = DataLoader(data_testing, batch_size = batch_size, shuffle = False) # num_workers = 0, pin_memory = True,    

        # define model, loss, optimizer and scheduler
        # model = mobilenet().to(device)

        # define model here
        if opt.models=="Mobilenet":
            net = mobilenet(freeze = opt.freeze).to(device)
            print("Model Selected: ", opt.models)
            # print(net)
        elif opt.models=="Resnet18":
            net = ResNet18(freeze = opt.freeze).to(device)
            print("Model Selected: ", opt.models)
            # print(net)
        elif opt.models=="Resnet34":
            net = ResNet34(freeze = opt.freeze).to(device)
            print("Model Selected: ", opt.models)
            # print(net)
        elif opt.models=="Resnet50":
            net = ResNet50(freeze = opt.freeze).to(device)
            print("Model Selected: ", opt.models)
            # print(net)
        else:
            print("Model Selected: ", opt.models)
            print('Invalid Model')
            exit()

        model = nn.DataParallel(net, device_ids = [0, 1]).to(device)
        loss_fn = nn.CrossEntropyLoss() # NLLLoss(weight=torch.tensor([imbalance_ratio, 1-imbalance_ratio])).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = wt) 
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                    max_lr = max_lr, # Upper learning rate boundaries in the cycle for each parameter group
                    steps_per_epoch = len(train_dataloader), # The number of steps per epoch to train for.
                    epochs = num_epoch, # The number of epochs to train for.
                    anneal_strategy = 'cos')


        # initiate variable for finding best epoch
        best_loss = float("inf") 
        best_epoch = 0 
        best_hsstss = 0
        learning_rate_values = []

        for t in range(num_epoch):
            
            t0 = time.time()
            train_loss, train_result = train_loop(train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=scheduler)
            test_loss, test_result = test_loop(test_dataloader,  model=model, loss_fn=loss_fn)
            table = confusion_matrix(test_result[:, 1], test_result[:, 0])
            HSS_score = HSS_multiclass(table)
            TSS_score = TSS_multiclass(table)
            F1_score = f1_score(test_result[:, 1], test_result[:, 0], average='macro')
            
            # time consumption and report R-squared values.
            duration = (time.time() - t0)/60
            print(f'Epoch {t+1}: Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, HSS: {HSS_score:.4f}, TSS: {TSS_score:.4f}, F1: {F1_score:.4f}, Duration(min):  {duration:.2f}')
                                        
            # trace score and predictions
            actual_lr = optimizer.param_groups[0]['lr']
            training_result.append([t, p, test_p, actual_lr, wt, train_loss, test_loss, HSS_score, TSS_score, F1_score, duration])
            torch.cuda.empty_cache()

            check_hsstss = (HSS_score * TSS_score)**0.5
            if best_hsstss < check_hsstss:
                best_hsstss = check_hsstss
                best_epoch = t+1
                best_loss = test_loss

                PATH = save_path + f"regmodel/{model_name}_{channel_tag}_freeze{opt.freeze}_20102024_train{p[0]}{p[1]}{p[2]}_test{test_p}.pth"
            # save model
                torch.save({
                        'epoch': t,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training loss': train_loss,
                        'testing loss' : test_loss,
                        'HSS_test' : HSS_score,
                        'TSS_test' : TSS_score
                        }, PATH)
                
                # save prediction array
                log_path = save_path + 'log/' + f'{model_name}_{channel_tag}_freeze{opt.freeze}_20102024_train{p[0]}{p[1]}{p[2]}_test{test_p}_' + \
                    f'lr{-math.log10(actual_lr):.1f}_decayval'
                
                if wt != 0:
                    log_path += f'{-math.log10(wt):.1f}.npy'
                else:
                    log_path += '0.npy'
                with open(log_path, 'wb') as f:
                    train_log = np.save(f, train_result)
                    test_log = np.save(f, test_result)

    training_result.append([f'Hyper parameters: batch_size: {batch_size}, number of epoch: {num_epoch}, initial learning rate: {lr}, decay value: {wt}'])

    # save the results
    #print("Saving the model's result")
    df_result = pd.DataFrame(training_result, columns=['Epoch', 'Train_p', 'Test_p', 
                                                        'learning rate', 'weight decay', 'Train_loss', 'Test_loss',
                                                        'HSS', 'TSS', 'F1_macro', 'Training-testing time(min)'])
    date = datetime.datetime.now()
    total_save_path = save_path + f'{model_name}_{channel_tag}_freeze{opt.freeze}_{date.year}{date.month:02d}_result_CV_lr{-math.log10(lr):.1f}_wt'
    if wt != 0:
        total_save_path += f'{-math.log10(wt):.1f}.csv'
    else:
        total_save_path += '0.csv'
    print('Save file here:', total_save_path)
    df_result.to_csv(total_save_path, index = False) 
        
print("Done!")