Command Example,

For cross validation with four partitions (tri-month partitioning)  
python -m Multi_imagery_SFPred.Main_CV --models Mobilenet --data All  

For validation, partition 1 & 2: training, parition 3: calibration, partition 4: testing  
python -m Multi_imagery_SFPred.cp_training --models Mobilenet --data All  

For calibration on partition 3,  
python -m Multi_imagery_SFPred.cp_cal --models Mobilenet --data All


### [Arguments]   
"--epochs": type = int, default = 4  
    number of epochs  
"--batch_size": type = int, default = 64
    batch size  
"--lr": type = float, default = 1e-6  
    learning rate    
"--weight_decay": type = list, default = [0, 1e-4], 
    regularization parameter    
"--max_lr": type = float, default = 1e-2  
    MAX learning rate  
"--models": type = str, default = 'Mobilenet'  
    Enter Model: Mobilenet, Resnet34, Resnet50
"--data": type = str, default = 'EUV304'   
    Enter Data source: EUV304, HMI-CTnuum, HMI-Mag, All  

### [Abstract]
In this project, we implement a binary classification task using multi-imagery datasets. Existing research has primarily focused on single-channel images for full-disk solar flare prediction. Interestingly, these studies have shown that simpler models, such as AlexNet, often outperform more complex ones. However, there may be untapped information in the solar surface or atmosphere that single-channel images fail to capture. We hypothesize that utilizing multi-channel imagery can address data insufficiency issues in deep learning, allowing more complex models to achieve higher performance.