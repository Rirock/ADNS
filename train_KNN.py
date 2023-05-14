import os
import argparse
import time
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data as Data 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

device = torch.device('cuda:0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--train_model", default="KNN", type=str, help="use model")
    parser.add_argument("-d", "--data_path", default="./Datasets/real_data/MAT\Leaf.mat", type=str, help="data path")
    parser.add_argument("-n", "--run_times", default=10, type=int, help="run times")
    parser.add_argument("-s", "--start_run_time", default=1, type=int, help="run times")

    args = parser.parse_args()

    model_name = getattr(args, 'train_model')
    data_path = getattr(args, 'data_path')
    run_times = getattr(args, 'run_times')
    start_run_time = getattr(args, 'start_run_time')

    path = os.getcwd()
    data_name = os.path.split(data_path)[-1].split(".")[0]

    # parameter
    epochs = 20000
    learning_rate = 0.001
    BATCH_SIZE = 1024*4
    model_save_path = "./models"

    # load data
    data = sio.loadmat(data_path)


    if "Datasets2" in data_path:
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


    log_path = os.path.join("./logs", data_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    train_num, input_size  = train_data.shape
    val_num, _  = val_data.shape
    test_num, _ = test_data.shape
    
    out_size = np.max(train_label)+1


    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    test_acc = []
    test_loss = []
    times = []
    auc = []
    st = time.time()

    loss_func = nn.CrossEntropyLoss()
    # log_path
    log_path = os.path.join(log_path, model_name+"_log.csv")

    # 训练
    clf = KNeighborsClassifier(5)
    clf.fit(train_data, train_label)
    # 预测
    y_pred = clf.predict(train_data)
    train_acc.append(sum(y_pred == np.squeeze(train_label)) / y_pred.shape[0])
    label = torch.from_numpy(train_label).squeeze().long()
    y_pred_onehot = torch.from_numpy(np.eye(out_size)[y_pred])
    loss = loss_func(y_pred_onehot, label)
    train_loss.append(loss.item())

    y_pred = clf.predict(test_data)
    test_acc.append(sum(y_pred == np.squeeze(test_label)) / y_pred.shape[0])
    label = torch.from_numpy(test_label).squeeze().long()
    y_pred_onehot = torch.from_numpy(np.eye(out_size)[y_pred])
    loss = loss_func(y_pred_onehot, label)
    test_loss.append(loss.item())
    
    print("ACC: %.2f%%" % (test_acc[0]*100))

    # AUC
    try:
        test_auc = roc_auc_score(test_label, y_pred, average="macro", multi_class='ovr')
    except:
        pred = np.eye(out_size)[y_pred]
        test_auc = roc_auc_score(test_label, pred, average="macro", multi_class='ovr')
    print("AUC:{:.4f}".format(test_auc))
    auc.append(test_auc)

    times.append(time.time() - st)


    log_list = [train_acc, train_loss, test_acc, test_loss, times, auc]
    log_list = list(map(list, zip(*log_list))) # 转置
    save_log = pd.DataFrame(log_list)
    save_log.to_csv(log_path, mode='w', header=False, index=False)
    print('mean train:', np.mean(train_acc), ' test:', np.mean(test_acc), ' auc:', np.mean(auc))
    print()