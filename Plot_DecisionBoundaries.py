# Generate decision boundary imgs

import os
import argparse
import time
import torch
import scipy.io as sio
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from model.DNM_models import *
from model.IC_model import *
from model.TransformerModel import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from get_result import get_data_by_gp
from model.ktree.custompackage.load_architecture import ktree_gen


device = torch.device('cuda:0')

# Custom Data Set Classes
class BatchDataset(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
def load_data(data_path, train=False):
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
        
    all_data = np.concatenate([train_data, val_data, test_data], axis=0)
    all_label = np.concatenate([train_label, val_label, test_label], axis=0)
    if train:
        return all_data, all_label, train_data, train_label, test_data, test_label
    else:
        return all_data, all_label


def Plot_DB(model_name_par, data_path, model_save_path="./models_all", log_path="./logs_all", img_save_path="./imgs", len_boundary=0.1, len_intensity=0.01, cmap="viridis"):
    # Plot_DecisionBoundaries
    data_name = os.path.split(data_path)[-1].split(".")[0]
    img_save_path = os.path.join(img_save_path, data_name)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    # load data
    all_data, all_label = load_data(data_path)

    # get result logs
    methods = [model_name_par]
    result_arr = get_data_by_gp(log_path, data_name, methods)[0]
    best_i = np.argmax(result_arr[:,2], axis=0)
    print("ACC:", np.max(result_arr[:,2], axis=0))
    
    hidden_size = 32
    if "DNM_multiple" in model_name_par or "ADNS" in model_name_par:
        M = int(model_name_par.split("_M")[-1])
        hidden_size = int(model_name_par.split("_")[-2])
        parts = model_name_par.split("_")
        model_name = "_".join(parts[:-2])
    elif "DNM_Linear" in model_name_par or "ADNM" in model_name_par:
        M = int(model_name_par.split("_M")[-1])
        parts = model_name_par.split("_")
        model_name = "_".join(parts[:-2])
    elif "MLP" in model_name_par or "Tran" in model_name_par or "ICFC" in model_name_par:
        hidden_size = int(model_name_par.split("_")[-1])
        parts = model_name_par.split("_")
        model_name = "_".join(parts[:-1])
    else:
        model_name = model_name_par

    train_num, input_size = all_data.shape
    out_size = np.max(all_label)+1

    if "MLP" in model_name or "IC" in model_name or "Transformer" in model_name:
        net = eval(model_name+"(input_size, hidden_size, out_size).to(device)")
        model_save_path = os.path.join(model_save_path, data_name+"_"+model_name+"_"+str(hidden_size)+"_"+str(best_i)+".pth")
    elif "DNM_multiple" in model_name or "ADNS" in model_name:
        net = eval(model_name+"(input_size, hidden_size, out_size, M).to(device)")
        model_save_path = os.path.join(model_save_path, data_name+"_"+model_name+"_"+str(hidden_size)+"_M"+str(M)+"_"+str(best_i)+".pth")
    elif "DNM_linear" in model_name or "ADNM" in model_name:
        net = eval(model_name+"(input_size, out_size, M).to(device)")
        model_save_path = os.path.join(model_save_path, data_name+"_"+model_name+"_M"+str(M)+"_"+str(best_i)+".pth")
    else:
        net = eval(model_name+"(input_size, out_size, M, device).to(device)")
        model_save_path = os.path.join(model_save_path, data_name+"_"+model_name+"_"+str(hidden_size)+"_M"+str(M)+"_"+str(best_i)+".pth")
   
    # load weight
    net.load_state_dict(torch.load(model_save_path))
    net.eval()
  
    # 准备数据
    X = all_data[:, :2]  # 只使用前两个特征
    y = all_label

    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - len_boundary, X[:, 0].max() + len_boundary
    y_min, y_max = X[:, 1].min() - len_boundary, X[:, 1].max() + len_boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, len_intensity),
                        np.arange(y_min, y_max, len_intensity))
    inin = np.c_[xx.ravel(), yy.ravel()]
    inin = torch.from_numpy(inin).to(device)
    inin_dataset = BatchDataset(inin)
    print(len(inin_dataset))
    inin_dataloader = Data.DataLoader(inin_dataset, batch_size=1024, shuffle=False)

    # 定义一个空的 tensor 存储所有的预测结果
    all_outputs = torch.empty(0)

    # 循环遍历每个 batch 进行预测
    for data in inin_dataloader:
        data = data.to(device).float()
        outputs = net(data)
        all_outputs = torch.cat((all_outputs, outputs.detach().cpu()))

    # Z = net(inin)
    Z = all_outputs
    Z = Z.detach().cpu().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.figure()
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.8)
 
    # 创建网格以绘制决策边界
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)

    # 添加标题和标签
    # plt.title('Decision Boundary')
    # plt.xlabel('Sepal Width')
    # plt.ylabel('Petal Length')
    plt.xticks([])
    plt.yticks([])

    # 显示图像
    # plt.show()

    # 保存图像
    plt.savefig(os.path.join(img_save_path,model_name+'.png'))
    plt.savefig(os.path.join(img_save_path,model_name+'.eps'), format='eps')


def Plot_DB_ktree(model_name_par, data_path, model_save_path="./models_all", log_path="./logs_all", img_save_path="./imgs", len_boundary=0.1, len_intensity=0.01, cmap="viridis"):
    # Plot_DecisionBoundaries
    data_name = os.path.split(data_path)[-1].split(".")[0]
    img_save_path = os.path.join(img_save_path, data_name)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    # load data
    all_data, all_label, train_data, train_label, test_data, test_labell = load_data(data_path, train=True)

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    train_data, target = train_data.to(device).float(), train_label.to(device).float()

    # get result logs
    model_name = model_name_par

    train_num, input_size = all_data.shape
    out_size = np.max(all_label)+1

    net = ktree_gen(ds='other', Activation="relu", Sparse=False, Input_order=None, Repeats=32, Padded=False, Input_size=input_size, Output_size=out_size, device=device)
    net.reset_parameters()
    net = net.to(device)
    net.train()
    train_acc = []
    train_loss = []
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(20000):
        output = net(train_data)
        output = torch.squeeze(output) 
        loss = loss_func(output, target.squeeze().long())
        optimizer.zero_grad()  # 梯度信息清空
        loss.backward()  # 反向传播获取梯度
        optimizer.step()  # 优化器更新
        
        if epoch % 100 == 0:
            print('Train Epoch: {} \tTrainLoss: {:.6f}'.format(epoch, loss.item()))
        if epoch == (20000 - 1):
            correct = 0
            total = 0
            predicted=torch.max(output.data, 1)[1] # predicted=torch.argmax(output, 1)
            target = torch.squeeze(target)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            train_acc.append(correct / total)
            train_loss.append(loss.item())

    print(train_acc)
    # load weight
    net.eval()
  
    # 准备数据
    X = all_data[:, :2] # 只使用前两个特征
    y = all_label

    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - len_boundary, X[:, 0].max() + len_boundary
    y_min, y_max = X[:, 1].min() - len_boundary, X[:, 1].max() + len_boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, len_intensity),
                        np.arange(y_min, y_max, len_intensity))
    inin = np.c_[xx.ravel(), yy.ravel()]
    inin = torch.from_numpy(inin).to(device)
    inin_dataset = BatchDataset(inin)
    print(len(inin_dataset))
    inin_dataloader = Data.DataLoader(inin_dataset, batch_size=1024, shuffle=False)

    # 定义一个空的 tensor 存储所有的预测结果
    all_outputs = torch.empty(0)

    # 循环遍历每个 batch 进行预测
    for data in inin_dataloader:
        data = data.to(device).float()
        outputs = net(data)
        all_outputs = torch.cat((all_outputs, outputs.detach().cpu()))

    # Z = net(inin)
    Z = all_outputs
    Z = Z.detach().cpu().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.figure()
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.8)
 
    # 创建网格以绘制决策边界
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)

    # 添加标题和标签
    # plt.title('Decision Boundary')
    # plt.xlabel('Sepal Width')
    # plt.ylabel('Petal Length')
    plt.xticks([])
    plt.yticks([])

    # 显示图像
    # plt.show()

    # 保存图像
    plt.savefig(os.path.join(img_save_path,model_name+'.png'))
    plt.savefig(os.path.join(img_save_path,model_name+'.eps'), format='eps')


def Plot_DB_SVM(model_name, data_path, img_save_path="./imgs", len_boundary=0.1, len_intensity=0.01, cmap="viridis"):
    # Plot_DecisionBoundaries
    data_name = os.path.split(data_path)[-1].split(".")[0]
    img_save_path = os.path.join(img_save_path, data_name)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    # load data
    all_data, all_label, train_data, train_label, test_data, test_label = load_data(data_path, True)
    train_num, input_size = all_data.shape
    out_size = np.max(all_label)+1

    model = svm.SVC(kernel='poly', C=1000)
    # 训练
    model.fit(train_data, train_label)
    print(model.score(train_data, train_label))

    # 预测
    y_pred = model.predict(test_data)
    acc = sum(y_pred == np.squeeze(test_label)) / y_pred.shape[0]    
    print("ACC: %.2f%%" % (acc*100))
  
    # 准备数据
    X = all_data[:, :2] # 只使用前两个特征
    y = all_label

    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - len_boundary, X[:, 0].max() + len_boundary
    y_min, y_max = X[:, 1].min() - len_boundary, X[:, 1].max() + len_boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, len_intensity),
                        np.arange(y_min, y_max, len_intensity))
    inin = np.c_[xx.ravel(), yy.ravel()]

    all_outputs = model.predict(inin)


    # Z = net(inin)
    Z = all_outputs
    # Z = Z.detach().cpu().numpy()
    # Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

   

    plt.figure()
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.8)
 
    # 创建网格以绘制决策边界
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)

    # 添加标题和标签
    # plt.title('Decision Boundary')
    # plt.xlabel('Sepal Width')
    # plt.ylabel('Petal Length')
    plt.xticks([])
    plt.yticks([])

    # 显示图像
    # plt.show()

    # 保存图像
    plt.savefig(os.path.join(img_save_path,model_name+'.png'))
    plt.savefig(os.path.join(img_save_path,model_name+'.eps'), format='eps')

def Plot_DB_KNN(model_name, data_path, img_save_path="./imgs", len_boundary=0.1, len_intensity=0.01, cmap="viridis"):
    # Plot_DecisionBoundaries
    data_name = os.path.split(data_path)[-1].split(".")[0]
    img_save_path = os.path.join(img_save_path, data_name)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    # load data
    all_data, all_label, train_data, train_label, test_data, test_label = load_data(data_path, True)
    train_num, input_size = all_data.shape
    out_size = np.max(all_label)+1

    clf = KNeighborsClassifier(5)
    clf.fit(train_data, train_label) # 训练
    # 预测
    y_pred = clf.predict(test_data)
    acc = sum(y_pred == np.squeeze(test_label)) / y_pred.shape[0]    
    print("ACC: %.2f%%" % (acc*100))
  
    # 准备数据
    X = all_data[:, :2] # 只使用前两个特征
    y = all_label

    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - len_boundary, X[:, 0].max() + len_boundary
    y_min, y_max = X[:, 1].min() - len_boundary, X[:, 1].max() + len_boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, len_intensity),
                        np.arange(y_min, y_max, len_intensity))
    inin = np.c_[xx.ravel(), yy.ravel()]

    all_outputs = clf.predict(inin)

    # Z = net(inin)
    Z = all_outputs
    # Z = Z.detach().cpu().numpy()
    # Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.figure()
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.8)
 
    # 创建网格以绘制决策边界
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)

    # 添加标题和标签
    # plt.title('Decision Boundary')
    # plt.xlabel('Sepal Width')
    # plt.ylabel('Petal Length')
    plt.xticks([])
    plt.yticks([])

    # 显示图像
    # plt.show()

    # 保存图像
    plt.savefig(os.path.join(img_save_path,model_name+'.png'))
    plt.savefig(os.path.join(img_save_path,model_name+'.eps'), format='eps')

if __name__ == "__main__":
    data_dir = "./Datasets/synthetic/MAT/"
    data_names = os.listdir(data_dir)
    model_name = "MLP_32"  # MLP_32 DNM_Linear_M20 DNM_multiple_32_M12   TransformerAttentionModel_32 ICFC ktree_gen
    
    model_save_path = "./save_models" # models_all models models_TransformerEmbeddingModel_nd_n8
    log_path = "./logs" # logs_all logs logs_TransformerEmbeddingModel_nd_n8

    img_save_path="./imgs"

    print(data_names)

    # 定义颜色
    # cmap = colors.ListedColormap(['#222449','#414C87','#9CB3D4'])#,'#ECEDEB','#AF5A76']) # 神里
    # cmap = colors.ListedColormap(['#E7BD39', '#47855A', '#384871']) # 香菱 '#AB4F3F',
    # cmap = colors.ListedColormap(['#AA4F23', '#2F2323', '#6C627A']) 

    plot_one_index = 4    # False
    for i in range(len(data_names)): 
        
        index = plot_one_index if plot_one_index else i
        
        data_name = data_names[index]    # XOR_Problem Moons Concentric Three_Spirals
        data_path = data_dir + data_name  
        
        if not os.path.exists(img_save_path):
            os.mkdir(img_save_path)

        if "SVM" in model_name:
            Plot_DB_SVM(model_name, data_path, img_save_path=img_save_path, len_boundary=0.1, len_intensity=0.005)
        elif "KNN" in model_name:
            Plot_DB_KNN(model_name, data_path, img_save_path=img_save_path, len_boundary=0.1, len_intensity=0.005)
        elif "ktree" in model_name:
            Plot_DB_ktree(model_name, data_path, model_save_path=model_save_path, log_path=log_path, len_boundary=0.1, len_intensity=0.005)
        else:
            Plot_DB(model_name, data_path, model_save_path=model_save_path, log_path=log_path, len_boundary=0.1, len_intensity=0.005)
            # Plot_DB(model_name, data_path, img_save_path=img_save_path, len_boundary=0.1, len_intensity=0.005)
        
        if plot_one_index:
            break
