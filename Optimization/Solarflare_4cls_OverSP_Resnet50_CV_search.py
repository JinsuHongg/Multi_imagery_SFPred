# basic package
import os
import math
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# pytorch package
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
#from pytorch_forecasting.metrics import QuantileLoss

# predefined class
from Multi_imagery_SFPred.Training import SolarFlSets, HSS, TSS, F1Pos, train_loop, test_loop, oversample_func
from Multi_imagery_SFPred.Models import mobilenet, ResNet34, ResNet50

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
img_dir = '/workspace/data/hetero_data/hmi/continuum/'
save_path = '/workspace/Project/Multi_imagery_SFPred/Results/CV/'
file_path = '/workspace/Project/Multi_imagery_SFPred/label_1h_bin/'

# Settings
model_name = 'Resnet50'
datasource = 'Mag'

# hyper-parameter 
batch_size = 16
num_epoch = 30
learning_rate = [1e-7, 5e-7, 5e-6, 1e-6, 5e-5, 1e-5]
decay_val = [0, 1e-5, 1e-4, 1e-3]

print(f'Hyper parameters: batch_size: {batch_size}, number of epoch: {num_epoch}, learning rate: {learning_rate}, decay value: {decay_val}')


'''
[ Grid search start here ] 
- Be careful with  result array, model, loss, and optimizer
- Their position matters

'''
'''
Define dataset here! 
'''
# Cross-validatation with optimization ( total = 4folds X Learning rate sets X weight decay sets )

for lr in learning_rate:
    for wt in decay_val:
        for i in range(0, 4):
            
            '''
            [ Grid search start here ] 
            - Be careful with  result array, model, loss, and optimizer
            - Their position matters

            '''

            p = [1, 2, 3, 4]
            test_p = p.pop(i)

            # Define dataset here! 
            train_list = [f'4image_GOES_classification_Partition{p[0]}.csv', 
                        f'4image_GOES_classification_Partition{p[1]}.csv', 
                        f'4image_GOES_classification_Partition{p[2]}.csv']
            
            test_file = f'4image_GOES_classification_Partition{test_p}.csv'
            
                    
            print('--------------------------------------------------------------------------------')
            print(f'Train: ({p[0]}, {p[1]}, {p[2]}), Test: {test_p}')
            print(f"learning rate: {lr:.1e}, decay value: {wt:.1e}")
            print('--------------------------------------------------------------------------------')

            # train set
            df_train = pd.DataFrame([], columns = ['Timestamp', 'goes_class', 'goes_class_num'])
            for partition in train_list:
                d = pd.read_csv(file_path + partition)
                df_train = pd.concat([df_train, d])

            # test set and calibration set
            df_test = pd.read_csv(file_path + test_file)

            # training data loader
            # over/under sampling
            NF_ins = SolarFlSets(annotations_df = df_train[df_train['goes_class_num'] == 0], 
                                img_dir = img_dir, normalization = True) # base target value
            base = len(NF_ins)

            FL_ins, FL_len = oversample_func(base_number = base, df = df_train[df_train['goes_class_num'] == 1], 
                                                 img_dir = img_dir, rstate = None, norm = True)

            final_set = [NF_ins]
            final_set.extend(FL_ins)

            print(f'Num NF instances: {base}, Num FL instances: {FL_len}')

            # validation data loader
            data_training = ConcatDataset(final_set)
            data_testing = SolarFlSets(annotations_df = df_test, img_dir = img_dir, normalization = True)
            train_dataloader = DataLoader(data_training, batch_size = batch_size, shuffle = True)
            test_dataloader = DataLoader(data_testing, batch_size = batch_size, shuffle = False)     

            # create result list and array 
            training_result = []
            lable_train_arr = np.empty((0,4), float)
            lable_test_arr = np.empty((0,4), float) 

            # define model, loss, optimizer and scheduler
            model = ResNet50().to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wt) 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5)

            # initiate variable for finding best epoch
            # best_loss = float("inf") 
            # best_epoch = 0 
            # best_hsstss = 0

            for t in range(num_epoch):
                
                t0 = time.time()
                train_loss, train_result = train_loop(train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=scheduler)
                test_loss, test_result = test_loop(test_dataloader,  model=model, loss_fn=loss_fn)
                table = confusion_matrix(test_result[:,1], test_result[:, 0])
                HSS = HSS(table)
                TSS = TSS(table)
                F1 = F1Pos(table)
                
                # time consumption and report R-squared values.
                duration = (time.time() - t0)/60
                print(f'Epoch {t+1}: Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, HSS: {HSS:.4f}, TSS: {TSS:.4f}, F1: {F1}, Duration(min):  {duration:.2f}')
                                            
                # trace score and predictions
                training_result.append([t, p, test_p, lr, wt, train_loss, test_loss, HSS, TSS, F1, duration])
                torch.cuda.empty_cache()

                # check_hsstss = (HSS + TSS)/2
                # if best_hsstss < check_hsstss:
                #     best_hsstss = check_hsstss
                #     best_epoch = t+1
                #     best_loss = test_loss

                #     PATH = save_path + f"/regmodel/Mobilenet_HMI_4cls_over_20102018_train{p[0]}-{p[1]}_test{test_p}.pth"
                # # save model
                #     torch.save({
                #             'epoch': t,
                #             'model_state_dict': model.state_dict(),
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'training loss': train_loss,
                #             'testing loss' : test_loss,
                #             'HSS_test' : HSS,
                #             'TSS_test' : TSS
                #             }, PATH)
                    
                #     # save prediction array
                #     log_path = save_path + '/log/' + f'Mobilenet_HMI_4cls_over_20102018_train{p[0]}-{p[1]}_test{test_p}_lr{-math.log10(lr):.1f}_decayval'
                #     if wt != 0:
                #         log_path += f'{-math.log10(wt):.1f}.npy'
                #     else:
                #         log_path += '0.npy'
                #     with open(log_path, 'wb') as f:
                #         train_log = np.save(f, train_result)
                #         test_log = np.save(f, test_result)

        training_result.append([f'Hyper parameters: train: {p}, test: {test_p}, batch_size: {batch_size}, number of epoch: {num_epoch}, learning rate: {lr}, decay value: {wt}'])

        # save the results
        #print("Saving the model's result")
        df_result = pd.DataFrame(training_result, columns=['Epoch', 'Train_p', 'Test_p', 
                                                           'learning rate', 'weight decay', 'Train_loss', 'Test_loss',
                                                           'HSS', 'TSS', 'Training-testing time(min)'])
        
        total_save_path = save_path + f'{model_name}_{datasource}_bin_over_20102024_result_CV_lr{-math.log10(lr):.1f}_wt'
        if wt != 0:
            total_save_path += f'{-math.log10(wt):.1f}.csv'
        else:
            total_save_path += '0.csv'
        print('Save file here:', total_save_path)
        df_result.to_csv(total_save_path, index = False) 
        
print("Done!")