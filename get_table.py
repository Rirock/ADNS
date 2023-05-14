# Generate latex tables

import matplotlib.pyplot as plt
import numpy as np
import os

from scipy import stats
from get_result import get_data_by_gp

def get_p_value(arrA, arrB):
    # get p_value by arr
    a = np.array(arrA)
    b = np.array(arrB)
    t, p = stats.ttest_ind(a,b)
    return p

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

path_use = 0
path_dict = ["./logs"]
path = path_dict[path_use]

methods = ["ADNS_32_M20", "MLP_32", "DNM_Linear_M20", "ktree_gen", "ICFC_32", "KNN", "SVM", "TransformerEmbeddingModel_32"]
methods2 = ["ADNS", "MLP(32)", "MDNN", "k-trees", "ICFC(32)", "KNN", "SVM", "Transformer"]

# patterns = ["Train Acc", "Train Loss", "Test Acc", "Test Loss"]
patterns = ["Auc"]
patterns_num = -1

# 获取数据
data_names = os.listdir(path)
data_names = []
data_path_list = ["./Datasets/real_data/MAT", "./Datasets2", "./Datasets/synthetic/MAT"]

for data_path in data_path_list:
    file_names = os.listdir(data_path)
    for file_name in file_names:
        data_names.append(file_name.split(".")[0])

print(data_names)
data_keys = []
for data_name in data_names:
    # dataname = data_name[:-5]
    dataname = data_name
    # if "EW" in dataname:
    #     dataname = dataname[:-2]
    data_keys.append(dataname)
print(data_keys)

# 创建文件
f = open('table.txt', 'w')
# 输出表格第一行
print("\hline", file=f)
for m in methods2:
    print("& "+m, end="", file=f)
print("\\\\\hline", file=f)
# f.close()
# p_value = 0
all_result_list = []
jsnum = [0 for _ in range(len(methods))]
num_2 = 0
for data_i in range(len(data_names)):  # 按数据集输出每一行
    result_list = get_data_by_gp(path, data_names[data_i], methods)  # 获取所有methods的数据----result[方法编号][总测试次数, 指标]
    all_result_list.append(result_list)
    mean_q = []  # mean_q[方法编号][指标编号]
    for i_m in range(len(methods)):  # 计算每个method的10次结果平均mean_q
        d_mean = np.mean(result_list[i_m], axis=0)
        mean_q.append(d_mean) 
    mean_q = np.array(mean_q) 
    mean_q = mean_q[:, patterns_num]  # 取每个算法的平均ACC
    winner = np.where(mean_q==np.amax(mean_q))[0]  # 取最优算法们的编号
    if mean_q[-1] == 0 : num_2 += 1
    for n in winner:
        jsnum[n] += 1
    # best_m = 0 # 该指标最优的算法
    # maxOmin = mean_q[0][patterns_num]
    # for i_m in range(1, len(methods)):
    #     if maxOmin<mean_q[i_m][patterns_num]: # find max
    #         maxOmin = mean_q[i_m][patterns_num]
    #         best_m = i_m

    # print
    print('F'+str(data_i+1)+'\t', end='',file=f) # 打印编号
    for i_m in range(len(methods)):

        d_mean = np.mean(result_list[i_m], axis=0)[patterns_num]
        d_std = np.std(result_list[i_m], axis=0)[patterns_num]

        if i_m in winner:
            print("& \\textbf{{{:.2f} $\pm$ {:.2f}}} (\%) \t".format(d_mean*100, d_std*100), end='',file=f)
        elif d_mean == 0:
            print("& {} \t".format("-"), end='',file=f)
        else:
            print("& {:.2f} $\pm$ {:.2f} (\%) \t".format(d_mean*100, d_std*100), end='',file=f)
    
    if data_i == len(data_names)-1:
        print('\\\\\hline',file=f)
    else:
        print('\\\\',file=f)
# all_result_list = np.array(all_result_list)

print("Mean",file=f)
for i_m in range(len(methods)):
    sum_mean = []
    for rl in all_result_list:
        d_mean = np.mean(rl[i_m], axis=0)[patterns_num]
        # d_std = np.std(result_list[i_m], axis=0)[patterns_num]
        sum_mean.append(d_mean)
        # sum_std += d_std
    sum_mean = np.array(sum_mean)
    d_mean = np.mean(sum_mean)
    if i_m==0:
        print("&\\textbf{{{:.2f}}}  (\%) \t".format(d_mean*100), end='',file=f)
    elif i_m != len(methods)-1:
        print("&{:.2f}  (\%) \t".format(d_mean*100), end='',file=f)
    else:
        d_mean = sum(sum_mean) / (len(data_names)-num_2)
        print("&{:.2f}  (\%) \t".format(d_mean*100), end='',file=f)

print('\\\\',file=f)

print("Median",file=f)
for i_m in range(len(methods)):
    sum_mean = []
    for rl in all_result_list:
        d_mean = np.mean(rl[i_m], axis=0)[patterns_num]
        # d_std = np.std(result_list[i_m], axis=0)[patterns_num]
        sum_mean.append(d_mean)
        # sum_std += d_std
    sum_mean = np.array(sum_mean)
    d_mean = np.median(sum_mean)
    if i_m==0:
        print("&\\textbf{{{:.2f}}}  (\%) \t".format(d_mean*100), end='',file=f)
    elif i_m != len(methods)-1:
        print("&{:.2f}  (\%) \t".format(d_mean*100), end='',file=f)
    else:
        sum_mean = sum_mean[sum_mean != 0]
        d_mean = np.median(sum_mean)
        print("&{:.2f}  (\%) \t".format(d_mean*100), end='',file=f)

print('\\\\',file=f)

print("Min ", end="", file=f)
for i_m in range(len(methods)):
    sum_mean = []
    for rl in all_result_list:
        d_mean = np.mean(rl[i_m], axis=0)[patterns_num]
        sum_mean.append(d_mean)
    sum_mean = np.array(sum_mean)
    d_min = np.min(sum_mean)
    if methods[i_m] == "ktree_gen" and d_min==0:
        sum_mean = np.where(sum_mean==0, 1, sum_mean)
        d_min = np.min(sum_mean)
    if i_m == 0:
        print("&\\textbf{{{:.2f}}}  (\%) \t".format(d_min*100), end='',file=f)
    else:
        print("&{:.2f}  (\%) \t".format(d_min*100), end='',file=f)
print('\\\\',file=f)

print("Max ", end="", file=f)
for i_m in range(len(methods)):
    sum_mean = []
    # sum_std = []
    for rl in all_result_list:
        d_mean = np.mean(rl[i_m], axis=0)[patterns_num]
        sum_mean.append(d_mean)
    sum_mean = np.array(sum_mean)
    d_max = np.max(sum_mean)
    if i_m == 0:
        print("&\\textbf{{{:.2f}}}  (\%) \t".format(d_max*100), end='',file=f)
    else:
        print("&{:.2f}  (\%) \t".format(d_max*100), end='',file=f)

print('\\\\',file=f)

print("Win ", end="", file=f)
for i_m in range(len(methods)):
    if i_m == 0:
        print("&\\textbf{{{}}}\t".format(jsnum[i_m]), end='',file=f)
    else:
        print("&{}\t".format(jsnum[i_m]), end='',file=f)


print('\\\\\hline',file=f)