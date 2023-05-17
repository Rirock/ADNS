# Generate scripts for training

import os

data_path_list = ["./Datasets/real_data/MAT", "./Datasets2", "./Datasets/synthetic/MAT"]

model_names = ["ADNM", "DNM_Linear", "DNM_multiple", "ADNS", "MLP", "ICFC", "KNN", "SVM", "TransformerEmbeddingModel", "ktree_gen"] 

def get_a_model_train_sh(model_name):
    # Generate training scripts by model_name
    run_times = 10
    hidden_size = 32
    DNM_M = 20

    save_path = "run_"+model_name+".sh"
    if os.path.exists(save_path):
        os.remove(save_path)

    for data_path in data_path_list:
        file_names = os.listdir(data_path)

        f = open(save_path, "a", encoding="utf-8")
        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            if "KNN" in model_name:
                f.write('python train_KNN.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
            elif "SVM" in model_name:
                f.write('python train_SVM.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
            elif "ktree_gen" in model_name:
                f.write('python train_ktree.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
            else:
                f.write('python train.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')

        f.close()

def get_all_model_train_sh(model_names):
    # 生成所有方法的脚本
    save_path = "run_all_model"+".sh"
    if os.path.exists(save_path):
        os.remove(save_path)
    run_times = 10
    hidden_size = 32
    DNM_M = 20
    
    for model_name in model_names: 

        for data_path in data_path_list:
            file_names = os.listdir(data_path)

            f = open(save_path, "a", encoding="utf-8")
            for file_name in file_names:
                file_path = os.path.join(data_path, file_name)
                if "KNN" in model_name:
                    f.write('python train_KNN.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
                elif "SVM" in model_name:
                    f.write('python train_SVM.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
                elif "ktree_gen" in model_name:
                    f.write('python train_ktree.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
                else:
                    f.write('python train.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')
            f.close()

def get_DNM_M_train_sh(model_name):
    # 生成不同M的训练脚本
    save_path = "run_all_DNM"+".sh"
    if os.path.exists(save_path):
        os.remove(save_path)
    run_times = 10
    hidden_size = 32
    DNM_M_LIST = range(2,21,2)
    for DNM_M in DNM_M_LIST:
        for data_path in data_path_list:
            file_names = os.listdir(data_path)
            f = open(save_path, "a", encoding="utf-8")
            for file_name in file_names:
                file_path = os.path.join(data_path, file_name)
                f.write('python train.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')
            f.close()


def get_demo_sh(model_names):
    # 生成DEMO脚本
    save_path = "run_demo"+".sh"
    if os.path.exists(save_path):
        os.remove(save_path)
    run_times = 1
    hidden_size = 32
    DNM_M = 20
    
    for model_name in model_names: 
        file_path = "./Datasets/synthetic/MAT/Three_Gaussians.mat"
        f = open(save_path, "a", encoding="utf-8")
        if "KNN" in model_name:
            f.write('python train_KNN.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
        elif "SVM" in model_name:
            f.write('python train_SVM.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
        elif "ktree_gen" in model_name:
            f.write('python train_ktree.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
        else:
            f.write('python train.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')
        f.close()



get_demo_sh(model_names)
# get_all_model_train_sh(model_names[3:])
# get_a_model_train_sh(model_names[0])
# get_DNM_M_train_sh(model_names[1])