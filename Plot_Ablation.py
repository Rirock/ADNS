# Generating plot for ablation experiments

import os
import numpy as np
import matplotlib.pyplot as plt

from get_result import get_data_by_method_mean

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

path = "./logs"
methods_names = ["ADNS_32_M", "ADNM_M", "DNM_multiple_32_M", "DNM_Linear_M"]
methods_names2 = ["ADNS", "No Mult", "No AEI", "No AEI&Mult"]
methods_names2 = ["ADNS", "P1", "P2", "P3"]
patterns_num = 2  # 2 is acc; -1 is auc

data_names = os.listdir(path)


x = range(2, 21, 2)
x = np.array(x)
print(x)
result_lists = []
y = [[] for _ in range(len(methods_names))]
for i, methods_name in enumerate(methods_names):
    result_list = []
    for m in x:
        method = methods_name+str(m)
        result = get_data_by_method_mean(path, data_names, method)
        # print(result)
        result_list.append(result)
        rr = np.mean(result, axis=0)[patterns_num]
        y[i].append(rr)
    result_lists.append(result_list)
print(y)
result_lists = np.array(result_lists)   # [模型名, M, 数据集, 指标]
print(result_lists.shape)

jsnum = np.zeros(len(x))
r0 = result_lists[0,:,:,patterns_num]
print(r0.shape)
for d in range(len(data_names)):
    r_m = r0[:, d]
    winner = np.where(r_m==np.amax(r_m))[0]
    for n in winner:
        jsnum[n] += 1


# Create canvas and subdrawings
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)

ax.plot(x, y[0], marker='o', color='#AB4F3F', linestyle="-", label=methods_names2[0], linewidth=2, markersize=8) # dashed   
ax.plot(x, y[1], marker='s', color='#47855A', linestyle="-", label=methods_names2[1], linewidth=2, markersize=8) 
ax.plot(x, y[2], marker='X', color='#384871', linestyle="-", label=methods_names2[2], linewidth=2, markersize=8) 
ax.plot(x, y[3], marker='^', color='#E7BD39', linestyle="-", label=methods_names2[3], linewidth=2, markersize=8)

# ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
ax.set_xticks(x) # Set the number of horizontal coordinates

# Set title and legend
plt.xlabel('M', fontsize=14)
# plt.ylabel('Acc', fontsize=14)
ax.set_title('ACC', fontsize=16) # Parameter M Comparison   Comparison of different modules
plt.legend(fontsize=10)

plt.show()
