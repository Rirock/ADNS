# Used to generate the dataset tables in the paper

import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

data_path_list = ["./Datasets/real_data/MAT", "./Datasets2", "./Datasets/synthetic/MAT"]

for data_path in data_path_list:
 
    file_names = os.listdir(data_path)

    for file_name in file_names:
        data_path_x = os.path.join(data_path, file_name)
        file_name = file_name.split(".")[0]
        if file_name.split("_")[-1] == "data":
            file_name = file_name[:-5]
        file_name = file_name.replace("_", " ")
        # load data
        data = sio.loadmat(data_path_x)


        if "Datasets2" in data_path_x:
            data1 = data['data']
            train_data = data1['train'][0][0]
            train_label = data1['trainLabel'][0][0][:,1]
            train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size = 0.2, random_state = 7)
            test_data = data1['test'][0][0]
            test_label = data1['testLabel'][0][0][:,1]
        else:
            train_data = data["Xtr"].T
            train_label = data["Ytr"].T-1
            train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size = 0.2, random_state = 7)
            test_data = data["Xtt"].T
            test_label = data["Ytt"].T-1


        train_num, input_size  = train_data.shape
        val_num, _  = val_data.shape
        test_num, _ = test_data.shape
        
        out_size = np.max(train_label)+1
        print("{},{},{},{},{},{}".format(file_name, input_size, out_size, train_num, val_num, test_num))
