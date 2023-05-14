# Generating plot for Hyperparameters experiments (M)

import os
import numpy as np
import matplotlib.pyplot as plt

from get_result import get_data_by_method_mean

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

path = "./logs_all"
methods_names = ["DNM_multiple_32_M", "DNM_Linear_M3_M", "DNM_multiple_DNM_32_M", "DNM_Linear_M"]
methods_names2 = ["MDNM", "P1", "P2", "P3"]
methods_names2 = ["MDNM", "No Mult", "No Attn", "No Attn&Mult"]
methods_names2 = ["Acc", "Auc", "No Attn", "No Attn&Mult"]
patterns_nums = [2,-1]
yy = []

for patterns_num in patterns_nums:
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

    yy.append(y)


print(yy[0][0])
print(yy[1][0])


y1 = yy[0][0]
y2 = yy[1][0]

# Creating a canvas
fig, ax = plt.subplots()
fig.set_size_inches(16, 6)

color1 = '#2E2C69'
color2 = '#8E8DC8'

# Draw bar chart
width = 0.8
ax.set_ylim(0.825, 0.9)
ax.bar(x - width/2, y1, width=width, color=color1, label='Acc')
ax.bar(x + width/2, y2, width=width, color=color2, label='Auc')
ax.axhline(y=0.8936909940924133, color=color1, linestyle='--')
ax.axhline(y=0.8815508388468921, color=color2, linestyle='--')


# Find the location of the maximum value
max_y1 = max(y1)
max_y2 = max(y2)
print(max_y1)
print(max_y2)
max_index_y1 = y1.index(max_y1)+1
print(max_index_y1)
max_index_y2 = y2.index(max_y2)+1
print(max_index_y2)

# Add specific values for each data point
for i, j in zip(x, y1):
    if max_index_y1*2 == i:
        ax.text(i, j, f"{j:.3f}", fontsize=12, ha='right', va='bottom', fontweight='bold') # Add a bold mark to the maximum value
    # else:
    #     ax.text(i, j, f"{j:.3f}", fontsize=14, ha='center', va='bottom')
for i, j in zip(x, y2):
    if max_index_y2*2 == i:
        ax.text(i, j, f"{j:.3f}", fontsize=12, ha='left', va='bottom', fontweight='bold') # Add a bold mark to the maximum value
    # else:
    #     ax.text(i, j, f"{j:.3f}", fontsize=14, ha='center', va='bottom')

# Set the number of horizontal coordinates
ax.set_xticks(x) 


# Set legend and title
plt.xlabel('M', fontsize=14)
plt.ylabel('Acc&Auc', fontsize=14)
# ax.set_title('Parameter M Comparison') # Parameter M Comparison   Comparison of different modules
ax.set_title('Effect of parameter M on ADNS', fontsize=16)
plt.legend(fontsize=12)

plt.show()