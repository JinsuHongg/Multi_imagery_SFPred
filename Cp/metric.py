import numpy as np

def compute_cov_length(label, predictionsets):
    if len(label) != len(predictionsets):
        print('The number of instances between two array is different. Please check your result.')
    else:
        size = len(label)
        
    count_ins = 0
    for i, each_label in enumerate(label):
        if predictionsets[i, each_label] != 0:
            count_ins += 1
    
    avg_cov = count_ins / size
    avg_length = np.sum(np.sum(predictionsets != 0, axis=1)) / size
        
    return avg_cov, avg_length

def conformity_score(cal_arr, test_arr, mode='CP', quantile=0.05, quantile_val=0):
    
    size_cal = len(cal_arr)
    quantile_cor = np.ceil((1+size_cal)*(1-quantile)) / size_cal # add calibaration set size effect
    label = cal_arr[:, 4].astype('int')
    softmax_result = cal_arr[:, 0:4]
    arr_result = np.copy(test_arr[:, 0:4]) 
    
    if mode == 'CP':
        arr_scores = 1 - cal_arr[range(size_cal), label.astype('int')]
        q_value = np.quantile(arr_scores, q=quantile_cor, method='higher') # find the quantile value in the calibarationset
        
        
        return arr_result*(arr_result >= 1-q_value), arr_scores, 1-q_value
    
    elif mode == 'AdpCP':

        Id_sort = np.argsort(softmax_result, axis=1)[:, ::-1]
        cal_arr_sort = np.take_along_axis(softmax_result, Id_sort, axis=1).cumsum(axis=1)
        arr_scores = np.take_along_axis(cal_arr_sort, Id_sort.argsort(axis=1), axis=1)[range(size_cal), label]
        q_value = np.quantile(arr_scores, q=quantile_cor, method='higher')
        
        Id_sort_test = test_arr[:, 0:4].argsort(1)[:,::-1]
        val_arr_sort = np.take_along_axis(test_arr[:, 0:4], Id_sort_test, axis=1).cumsum(axis=1)
        prediction_sets = np.take_along_axis(val_arr_sort <= q_value, Id_sort_test.argsort(axis=1), axis=1)
        
        # if APS has zero set, let the set have setsize of 1
        non_zero_count = np.count_nonzero(prediction_sets, axis=1)
        block_allzero = np.zeros((arr_result.shape[0], arr_result.shape[1]))
        for index, bool_val in enumerate(non_zero_count==0):
            if bool_val:
                max_index = np.argmax(arr_result[index, :])
                block_allzero[index, max_index] = np.max(arr_result[index, :])
        
        return (arr_result*prediction_sets + block_allzero), arr_scores, q_value