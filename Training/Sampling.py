import torchvision.transforms as transforms
from . import SolarFlSets


def oversample_func(base_number=1000, df=None, img_dir=None, rstate=1004, norm=True):
        
    # define transformations
    rotation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(-5,5)),
        transforms.ToTensor()
    ])

    hr_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ])

    vr_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor()
    ])  
    
    ori_dt = SolarFlSets(annotations_df = df, 
                      img_dir = img_dir, normalization = norm) 

    total_len = 0
    im_ratio = base_number/len(ori_dt) # sampling ratio

    rot_dt = SolarFlSets(annotations_df = df, 
                              img_dir = img_dir, transform = rotation, normalization = norm)
    hflip_dt = SolarFlSets(annotations_df = df, 
                                img_dir = img_dir, transform = hr_flip, normalization = norm) 
    vflip_dt = SolarFlSets(annotations_df = df, 
                                img_dir = img_dir, transform = vr_flip, normalization = norm) 

    sampling_list = [rot_dt, hflip_dt, vflip_dt]
    final_dtlist = [ori_dt]
    total_len += len(ori_dt)
            
    oversample_dt = [sampling_list[num_iter%3] for num_iter in range(int(im_ratio)-1)]
    
    # calculate the total lengh
    for num_iter in range(int(im_ratio)-1):
        total_len += len(sampling_list[num_iter%3])

    final_dtlist.extend(oversample_dt)

    trans_index = ((int(im_ratio) - 1) // 3) % 3
    if trans_index == 0: # add rotation transformation
        add_rot_dt = SolarFlSets(annotations_df = df, 
                                 img_dir = img_dir, transform = rotation, 
                                 num_sample = base_number - int(im_ratio)*len(ori_dt), 
                                 random_state = rstate,
                                 normalization = norm)
        final_dtlist.append(add_rot_dt)
        total_len += len(add_rot_dt)

    elif trans_index == 1: # add horizontal flip
        add_hflip_dt = SolarFlSets(annotations_df = df, 
                                   img_dir = img_dir, transform = hr_flip, 
                                   num_sample = base_number - int(im_ratio)*len(ori_dt), 
                                   random_state = rstate,
                                   normalization = norm)
        final_dtlist.append(add_hflip_dt)
        total_len += len(add_hflip_dt)

    elif trans_index == 2: # add vertical flip
        add_hflip_dt = SolarFlSets(annotations_df = df,
                                   img_dir = img_dir, transform = vr_flip, 
                                   num_sample = base_number - int(im_ratio)*len(ori_dt), 
                                   random_state = rstate,
                                   normalization = norm)
        final_dtlist.append(add_hflip_dt)
        total_len += len(add_hflip_dt)

    return final_dtlist, total_len