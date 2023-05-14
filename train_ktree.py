import os
import argparse
import time
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch.utils.data as Data 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model.ktree.custompackage.load_architecture import ktree_gen

device = torch.device('cuda:0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--train_model", default="ktree_gen", type=str, help="use model") # ICFC DNM_Linear
    parser.add_argument("-d", "--data_path", default="./Datasets2/Australia_data.mat", type=str, help="data path")
    parser.add_argument("-n", "--run_times", default=1, type=int, help="run times")
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
    model_save_path = "./save_models"
    l_path = "./logs"
    if not os.path.exists(l_path):
        os.mkdir(l_path)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

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


    log_path = os.path.join(l_path, data_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    train_num, input_size  = train_data.shape
    val_num, _  = val_data.shape
    test_num, _ = test_data.shape
    
    out_size = np.max(train_label)+1
    
    # if out_size>2:
    #     exit()

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    test_acc = []
    test_loss = []
    times = []
    auc = []

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    val_data = torch.from_numpy(val_data)
    val_label = torch.from_numpy(val_label)
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)
    
    data, target = train_data.to(device).float(), train_label.to(device).float()
    val_data, val_target = val_data.to(device).float(), val_label.to(device).float()
    test_data, test_target = test_data.to(device).float(), test_label.to(device).float()

    net = ktree_gen(ds='other', Activation="relu", Sparse=False, Input_order=None, Repeats=4, Padded=False, Input_size=input_size, Output_size=out_size, device=device)
    # net = ktree_gen(ds='other', Activation="None", Sparse=False, Input_order=None, Repeats=1, Padded=False, Input_size=input_size, device=device)
    log_path = os.path.join(log_path, model_name+"_log.csv")

    for run in range(run_times):
        st = time.time()
        best_val_acc = 0
        current_val_acc = best_val_acc
        best_val_loss = np.inf
        current_val_loss = best_val_loss
        no_improve = 0

        net.reset_parameters()
        net = net.to(device)
        net.train()
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            
            output = net(data)
            output = torch.squeeze(output) 
            loss = loss_func(output, target.squeeze().long())
            optimizer.zero_grad()  # 梯度信息清空
            loss.backward()  # 反向传播获取梯度
            optimizer.step()  # 优化器更新
            
            if epoch % 100 == 0:
                print('Train Epoch: {} \tTrainLoss: {:.6f}\tValLoss: {:.6f}\tValAcc: {:.6f}'.format(epoch, loss.item(), current_val_loss, current_val_acc))
            if epoch == (epochs - 1):
                correct = 0
                total = 0
                predicted=torch.max(output.data, 1)[1] # predicted=torch.argmax(output, 1)
                target = torch.squeeze(target)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                train_acc.append(correct / total)
                train_loss.append(loss.item())

            # val
            val_fit = net(val_data)
            val_fit = torch.squeeze(val_fit) 
            predicted=torch.max(val_fit.data, 1)[1] # 
            correct = (predicted == val_target.squeeze()).sum().item()
            current_val_acc = correct / val_num
            current_val_loss = loss_func(val_fit, val_target.squeeze().long())
            
            #val_acc.append(current_val_acc)

            if current_val_acc > best_val_acc or current_val_loss<best_val_loss:
                best_val_acc = current_val_acc
                best_val_loss = current_val_loss
                save_path = os.path.join(model_save_path, data_name+"_"+model_name+"_"+str(run)+".pth")
                torch.save(net.state_dict(), save_path)
                no_improve = 0
            elif no_improve > 5000:
                correct = 0
                total = 0
                predicted=torch.max(output.data, 1)[1] # predicted=torch.argmax(output, 1)
                target = torch.squeeze(target)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                train_acc.append(correct / total)
                train_loss.append(loss.item())
                break # 不提升停止训练
            else:
                no_improve += 1


            #print(run+1, ' train_acc:', train_acc[run], ' train_loss:', train_loss[run], ' test_acc:', test_acc[run], ' test_loss:', test_loss[run], ' time:', times[run])

        # Testing
        net.load_state_dict(torch.load(save_path))
        net.eval()
        test_fit = net(test_data)
        test_fit = torch.squeeze(test_fit) 
        predicted=torch.max(test_fit.data, 1)[1] # 
        correct = (predicted == test_target.squeeze()).sum().item()
        test_acc.append(correct / test_num)

        loss = loss_func(test_fit, test_target.squeeze().long())
        test_loss.append(loss.item())

        #save_path = os.path.join(model_save_path, data_name+"_"+model_name+"_"+str(run)+".pth")
        #torch.save(net.state_dict(), save_path)
        times.append(time.time() - st)

        # AUC
        y_pred = torch.argmax(test_fit, -1).cpu().numpy()
        try:
            test_auc = roc_auc_score(test_label, y_pred, average="macro", multi_class='ovr')
        except:
            pred = np.eye(out_size)[y_pred]
            test_auc = roc_auc_score(test_label, pred, average="macro", multi_class='ovr')
        # print("AUC:{:.4f}".format(test_auc))
        auc.append(test_auc)
        
        print(run+1, ' train_acc:', train_acc[run], ' train_loss:', train_loss[run], ' test_acc:', test_acc[run], ' test_loss:', test_loss[run], ' time:', times[run], ' auc:', auc[run])


    log_list = [train_acc, train_loss, test_acc, test_loss, times, auc]
    log_list = list(map(list, zip(*log_list))) # 转置
    save_log = pd.DataFrame(log_list)
    save_log.to_csv(log_path, mode='w', header=False, index=False)
    print('mean train:', np.mean(train_acc), ' test:', np.mean(test_acc))
    print()