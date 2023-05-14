import os
import csv
import numpy as np

def get_data_by_gp(path, data_name, methods):
    # 获取一个数据集中所有方法的结果 result[方法编号][总测试次数, 指标]
    result_list = []
    for method in methods:
        logDiri = os.path.join(path, data_name, method+"_log.csv")
        if os.path.exists(logDiri):
            csv_r = csv.reader(open(logDiri, "r"))
            r = []
            for i, num in enumerate(csv_r):
                r.append(num)
            r = np.array(r).astype(float)
            #rr = np.mean(r, axis=0)
            result_list.append(r)
        else:
            result_list.append(np.zeros([1,6]))
    return(result_list)

def get_data_by_method_mean(path, data_names, method):
    # 获取所有数据集中一个方法的结果 result[方法编号][总测试次数, 指标]
    result_list = []
    for data_name in data_names:
        logDiri = os.path.join(path, data_name, method+"_log.csv")
        csv_r = csv.reader(open(logDiri, "r"))
        r = []
        for i, num in enumerate(csv_r):
            r.append(num)
        r = np.array(r).astype(float)
        rr = np.mean(r, axis=0)
        result_list.append(rr)
    return(result_list)

if __name__ == "__main__":
    data_path = "./logs"    # Data set used
    log_path = "./logs"     # logs path
    file_names = os.listdir(data_path)
    model = False  # "DNM_Linear"  # Comparing the results of different M
    models = ["MLP_32", "DNM_Linear_M20", "ADNS_32_M12", "ICFC_32", "KNN", "SVM", "TransformerEmbeddingModel_32", "ktree_gen"]
    nameEnd = "_log.csv"
    pattern = 2  # -1 is AUC; 2 is ACC
     
    if model:  # Comparing the results of different M in DNM
        models = []
        for i in range(2, 21, 2):
            models.append(model+"_M"+str(i))
            
    folder_paths = []
    for folder in file_names:
        folder_path = os.path.join(log_path, folder)
        if os.path.isdir(folder_path):
            folder_paths.append(folder)

    result = []
    jsnum = [0 for _ in range(len(models))]
    r_sum = [0 for _ in range(len(models))]
    num_2 = 0
    for file_name in folder_paths:
        logDir = os.path.join(log_path, file_name.split(".")[0])
        print(file_name.split(".")[0])
        r_m = []
        
        for model in models:
            logDiri = os.path.join(logDir, model+nameEnd)
            if os.path.exists(logDiri):
                csv_r = csv.reader(open(logDiri, "r"))
                r = []
                for i, num in enumerate(csv_r):
                    r.append(num)
                r = np.array(r)
                r = r.astype(float)
                print("%-20s"%model, end=": \t")
                rr = np.mean(r, axis=0)
                for i in range(len(rr)):
                    print("%.3f"%rr[i], end=", ")
                print()
                r_m.append(rr[pattern])
            else:
                r_m.append(0)
                num_2 += 1
        print()
        r_sum = np.array(r_sum) + np.array(r_m)
        winner = np.where(r_m==np.amax(r_m))[0]
        print(winner)
        #n = np.argmax(r_m)
        for n in winner:
            jsnum[n] += 1
            print(models[n], end=", ")
        print()
        print()
        result.append(r_m)

    r_sum[:-1] = r_sum[:-1]/len(folder_paths)
    r_sum[-1] = r_sum[-1]/(len(folder_paths)-num_2)
    print(r_sum)
    print(jsnum)

