import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import fire
from scipy.stats import skew
import torch
from safetensors.torch import load_file,save_file
import os
import shutil
import yaml


def decode_module_name(name):
    layer = name.str.extract(r'layers\.(\d+)\.')[0]
    parameter = name.str.extract(r'(\w+_pack\.lora_[AB]|\w_proj\.lora_[AB])')[0]
    #parameter = name.str.extract(r'(\w+_pack\.\|\|\|\d+|o_proj\.\|\|\|\d+)')[0]
    return "Layer("+layer+") "+parameter

def refine_module_name(name):
    return name

def get_threshold(
        csv_file_path,
        metric: str = "1sd" #1sd,2sd,90,skew90
                  ):
    # Load the CSV file

    data = pd.read_csv(csv_file_path)

    # Extract layer number and parameter from the 'Module_Name' column
    data['Layer'] = data['Module_Name'].str.extract(r'layers\.(\d+)\.')[0]
    data['Parameter'] = data['Module_Name'].str.extract(r'(\w+_pack\.lora_[AB]|\w_proj\.lora_[AB])')[0]
    #data['Parameter'] = data['Module_Name'].str.extract(r'(\w+_pack\.\|\|\|\d+|\w_proj\.\|\|\|\d+)')[0]
    heatmap_data = data['Importance_Score']

    all_values = heatmap_data.values.flatten()

    print(np.sum(all_values != 0))

    overall_mean = all_values.mean()
    overall_variance = all_values.var()
    overall_skewness = skew(all_values)


    print("Overall Mean:", overall_mean)
    print("Overall Variance:", overall_variance)
    print("Overall Skewness:", overall_skewness)

    dic = {}

    # Mean + 1 Standard Deviation
    threshold_1sd = overall_mean + overall_variance

    # Mean + 2 Standard Deviations
    threshold_2sd = overall_mean + 2 * overall_variance

    # 90th Percentile
    threshold_90 = np.percentile(all_values, 90)


    above_threshold = all_values > threshold_1sd
    count_above = np.sum(above_threshold)
    indices_above = np.where(above_threshold)[0]
    dic["1sd"] = {
        "threshold":threshold_1sd,
        "count_above":count_above,
        "Module_above":data["Module_Name"][indices_above]
    }
    
    above_threshold = all_values >  threshold_2sd
    count_above = np.sum(above_threshold)
    indices_above = np.where(above_threshold)[0]
    dic["2sd"] = {
        "threshold":threshold_2sd,
        "count_above":count_above,
        "Module_above":data["Module_Name"][indices_above]
    }

    above_threshold = all_values > threshold_90
    count_above = np.sum(above_threshold)
    indices_above = np.where(above_threshold)[0]
    dic["90"] = {
        "threshold":threshold_90,
        "count_above":count_above,
        "Module_above":data["Module_Name"][indices_above]
    }
    
    median = np.median(all_values)
    mad = np.median(np.abs(all_values - median))


    z_scores = (all_values - median) / mad if mad != 0 else all_values - median


    skewness_factor = 1 + overall_skewness / 10
    weighted_scores = z_scores * all_values * skewness_factor


    threshold = np.percentile(weighted_scores, 90)


    above_threshold = weighted_scores > threshold
    count_above = np.sum(above_threshold)
    indices_above = np.where(above_threshold)[0]
    dic["skew90"] = {
        "threshold":threshold,
        "count_above":count_above,
        "Module_above":data["Module_Name"][indices_above]
    }


    return dic[metric],data["Module_Name"]

def process_file_path(dataset_id,service_begin_id,dataset_order,model_name,adherence_checkpoint,robustness_checkpoint,all_checkpoint):
    ipt_file_adherence = "./ipt_file/Importance4_Score_averaging_dataset_id_"+ str(dataset_id) + "_" + str(service_begin_id)+"-"+dataset_order[service_begin_id]+"-"+model_name+".csv"
    ipt_file_robustness = "./ipt_file/Importance4_Score_averaging_dataset_id_"+ str(dataset_id) + "_" + str(service_begin_id+1)+"-"+dataset_order[service_begin_id+1]+"-"+model_name+".csv"
    adherence_mix_weights_path = ""
    robustness_mix_weights_path = ""
    all_mix_weights_path = ""
    merged_weights_path = ""
    if model_name == "llama2":
        adherence_mix_weights_path = "./saves/llama2-7b/SqaudZen_extraction_adherence_mix_llama_epoch_1_lr_5e-5/"
        robustness_mix_weights_path = "./saves/llama2-7b/SqaudZen_extraction_robustness_mix_llama_epoch_1_lr_5e-5/"
        all_mix_weights_path = "./saves/llama2-7b/SqaudZen_extraction_all_mix_llama_epoch_1_lr_5e-5"
        merged_weights_path = "./saves/llama2-7b/merged/"
    else:
        adherence_mix_weights_path = "./saves/qwen1.5-14b/SqaudZen_extraction_adherence_mix_qwen_epoch_1_lr_5e-5/"
        robustness_mix_weights_path = "./saves/qwen1.5-14b/SqaudZen_extraction_robustness_mix_qwen_epoch_1_lr_5e-5/"
        all_mix_weights_path = "./saves/qwen1.5-14b/SqaudZen_extraction_all_mix_qwen_epoch_1_lr_5e-5/"
        merged_weights_path = "./saves/qwen1.5-14b/merged/"
    adherence_mix_weights_path = os.path.join(adherence_mix_weights_path,"checkpoint-"+str(adherence_checkpoint))
    robustness_mix_weights_path = os.path.join(robustness_mix_weights_path,"checkpoint-"+str(robustness_checkpoint))
    all_mix_weights_path = os.path.join(all_mix_weights_path,"checkpoint-"+str(all_checkpoint))
    merged_weights_path = os.path.join(merged_weights_path,str(adherence_checkpoint)+"|"+str(robustness_checkpoint)+"|"+str(all_checkpoint)+"|")
    return ipt_file_adherence,ipt_file_robustness,adherence_mix_weights_path,robustness_mix_weights_path,all_mix_weights_path,merged_weights_path
        


def group_modules(metric,ipt_file_adherence,ipt_file_robustness):
    adherence_dic,all_elements_adherence = get_threshold(csv_file_path=ipt_file_adherence,metric=metric)
    print(adherence_dic["threshold"],adherence_dic["count_above"],decode_module_name(adherence_dic["Module_above"]))
    robustness_dic,all_elements_robostness = get_threshold(csv_file_path=ipt_file_robustness,metric=metric)
    print(robustness_dic["threshold"],robustness_dic["count_above"],decode_module_name(robustness_dic["Module_above"]))

    assert len(all_elements_robostness)==len(all_elements_adherence)
    
    common_elements = adherence_dic["Module_above"][adherence_dic["Module_above"].isin(robustness_dic["Module_above"])].reset_index(drop=True)

    # Find elements unique to A while preserving order
    unique_adherence_dic = adherence_dic["Module_above"][~adherence_dic["Module_above"].isin(robustness_dic["Module_above"])].reset_index(drop=True)

    # Find elements unique to B while preserving order
    unique_robustness_dic = robustness_dic["Module_above"][~robustness_dic["Module_above"].isin(adherence_dic["Module_above"])].reset_index(drop=True)

    common_elements = [refine_module_name(item) for item in common_elements.tolist()]
    unique_adherence_dic = [refine_module_name(item) for item in unique_adherence_dic.tolist()]
    unique_robustness_dic = [refine_module_name(item) for item in unique_robustness_dic.tolist()]
    # Display the results
    return common_elements,unique_adherence_dic,unique_robustness_dic,all_elements_adherence.tolist()



#print(common_elements,unique_adherence_dic,unique_robustness_dic,all_elements)


def lora_merging(merged_path, all_elements, common_elements, unique_adherence_elements, unique_robustness_elements, lora_adherence_path, lora_robustness_path, lora_classification_path,model_name):
    checkpoint_adherence = load_file(os.path.join(lora_adherence_path,"adapter_model.safetensors"))
    checkpoint_robustness = load_file(os.path.join(lora_robustness_path,"adapter_model.safetensors"))
    checkpoint_classification = load_file(os.path.join(lora_classification_path,"adapter_model.safetensors"))
    #weighted_weights = {}
    weighted_weights = {key: torch.zeros_like(param).to('cuda:0') for key, param in checkpoint_adherence.items()}
    print(checkpoint_adherence.keys(),checkpoint_robustness.keys(),checkpoint_classification.keys())
    for key in all_elements:
        rank_r = int(key.split("|||")[1])
        ori_loraA_name = key.split("|||")[0] + "lora_A.weight"
        ori_loraB_name = key.split("|||")[0] + "lora_B.weight"
        
        tensor_adherence_loraA = checkpoint_adherence[ori_loraA_name].to('cuda:0')
        tensor_robustness_loraA = checkpoint_robustness[ori_loraA_name].to('cuda:0')
        tensor_classification_loraA = checkpoint_classification[ori_loraA_name].to('cuda:0')
        
        tensor_adherence_loraB = checkpoint_adherence[ori_loraB_name].to('cuda:0')
        tensor_robustness_loraB = checkpoint_robustness[ori_loraB_name].to('cuda:0')
        tensor_classification_loraB = checkpoint_classification[ori_loraB_name].to('cuda:0')

        print(tensor_classification_loraB.shape,tensor_robustness_loraB.shape,tensor_adherence_loraB.shape)
        if key in common_elements: 
            weighted_weights[ori_loraA_name][rank_r,:] = 0.2*tensor_classification_loraA[rank_r,:] + \
                0.4*tensor_adherence_loraA[rank_r,:] + 0.4*tensor_robustness_loraA[rank_r,:]
            weighted_weights[ori_loraB_name][:,rank_r] = 0.2*tensor_classification_loraB[:,rank_r] + \
                0.4*tensor_adherence_loraB[:,rank_r] + 0.4*tensor_robustness_loraB[:,rank_r]

        elif key in unique_adherence_elements:
            weighted_weights[ori_loraA_name][rank_r,:] = 0.2*tensor_classification_loraA[rank_r,:] + \
                0.8*tensor_adherence_loraA[rank_r,:]
            weighted_weights[ori_loraB_name][:,rank_r] = 0.2*tensor_classification_loraB[:,rank_r] + \
                0.8*tensor_adherence_loraB[:,rank_r]
            
        elif key in unique_robustness_elements:
            weighted_weights[ori_loraA_name][rank_r,:] = 0.2*tensor_classification_loraA[rank_r,:] + \
                0.8*tensor_robustness_loraA[rank_r,:]
            weighted_weights[ori_loraB_name][:,rank_r] = 0.2*tensor_classification_loraB[:,rank_r] + \
                0.8*tensor_robustness_loraB[:,rank_r]
        else:
            weighted_weights[ori_loraA_name][rank_r,:] = 0.8*tensor_classification_loraA[rank_r,:] + \
                0.1*tensor_adherence_loraA[rank_r,:] + 0.1*tensor_robustness_loraA[rank_r,:]
            weighted_weights[ori_loraB_name][:,rank_r] = 0.8*tensor_classification_loraB[:,rank_r] + \
                0.1*tensor_adherence_loraB[:,rank_r] + 0.1*tensor_robustness_loraB[:,rank_r]

        # weighted_weights[key] = 0.5*tensor1+0.5*tensor2
        
    save_path = merged_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert len(weighted_weights) == len(checkpoint_classification)
    save_file(weighted_weights, os.path.join(save_path,"adapter_model.safetensors"))


    source_path = os.path.join(lora_adherence_path, "adapter_config.json")
    destination_path = os.path.join(save_path, "adapter_config.json")
    yaml_path = os.path.join(save_path,"inference_api.yaml")

    if model_name == "llama2":
        data = {
        'model_name_or_path': "",
        'template': "llama2",
        'finetuning_type': "lora"
    }
    else:
        data = {
        'model_name_or_path': "",
        'template': "qwen",
        'finetuning_type': "lora"
    }
    data["adapter_name_or_path"] = save_path
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)

    shutil.copy(source_path, destination_path)



def lora_merging_matrix(merged_path, all_elements, common_elements, unique_adherence_elements, unique_robustness_elements, lora_adherence_path, lora_robustness_path, lora_classification_path,model_name):
    checkpoint_adherence = load_file(os.path.join(lora_adherence_path,"adapter_model.safetensors"))
    checkpoint_robustness = load_file(os.path.join(lora_robustness_path,"adapter_model.safetensors"))
    checkpoint_classification = load_file(os.path.join(lora_classification_path,"adapter_model.safetensors"))
    weighted_weights = {}

    for key in checkpoint_classification:
        tensor_adherence = checkpoint_adherence[key].to('cuda:0')
        tensor_robustness = checkpoint_robustness[key].to('cuda:0')
        tensor_classification = checkpoint_classification[key].to('cuda:0')
        #print(checkpoint_present[key])
        #print(checkpoint_previous[key])
        #print(0.5*tensor1+0.5*tensor2)
        
        if key in common_elements:
            weighted_weights[key] = 0.2*tensor_classification + 0.4*tensor_adherence + 0.4*tensor_robustness
        elif key in unique_adherence_elements:
            weighted_weights[key] = 0.2*tensor_classification + 0.8*tensor_adherence
        elif key in unique_robustness_elements:
            weighted_weights[key] = 0.2*tensor_classification + 0.8*tensor_robustness
        else:
            weighted_weights[key] = 0.8*tensor_classification + 0.1*tensor_adherence + 0.1*tensor_robustness
        # weighted_weights[key] = 0.5*tensor1+0.5*tensor2
        
    save_path = merged_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert len(weighted_weights) == len(checkpoint_classification)
    save_file(weighted_weights, os.path.join(save_path,"adapter_model.safetensors"))


    source_path = os.path.join(lora_adherence_path, "adapter_config.json")
    destination_path = os.path.join(save_path, "adapter_config.json")

    yaml_path = os.path.join(save_path,"inference_api.yaml")

    if model_name == "llama2":
        data = {
        'model_name_or_path': "",
        'template': "llama2",
        'finetuning_type': "lora"
    }
    else:
        data = {
        'model_name_or_path': "",
        'template': "qwen",
        'finetuning_type': "lora"
    }
    data["adapter_name_or_path"] = save_path
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)

    shutil.copy(source_path, destination_path)




def train(
    dataset_id: int = 0, # 1 - 5  
    service_begin_id: int = 0, # 
    model_name:str = "llama2",
    adherence_checkpoint:int = 5000,
    robustness_checkpoint:int = 5000,
    all_checkpoint:int = 5000,
):
    
    dataset_order = ["adherence_0","robustness_0"]
    
    ipt_file_adherence, \
    ipt_file_robustness,    \
    adherence_mix_weights_path, \
    robustness_mix_weights_path,    \
    all_mix_weights_path,merged_weights_path = process_file_path(dataset_id,service_begin_id,dataset_order,model_name,adherence_checkpoint,robustness_checkpoint,all_checkpoint)


    common_elements,unique_adherence_dic,unique_robustness_dic,all_elements = group_modules("1sd",ipt_file_adherence,ipt_file_robustness)
    print(f"common elements: {len(common_elements)}; {common_elements}")
    print(f"unique_adherence: {len(unique_adherence_dic)}; {unique_adherence_dic}")
    print(f"unique_robustness: {len(unique_robustness_dic)}; {unique_robustness_dic}")

    lora_merging_matrix(merged_path=merged_weights_path + "lora2_test",
             common_elements=common_elements,
             unique_adherence_elements=unique_adherence_dic,
             unique_robustness_elements=unique_robustness_dic,
             all_elements=all_elements,
             lora_adherence_path=adherence_mix_weights_path,
             lora_robustness_path=robustness_mix_weights_path,
             lora_classification_path=all_mix_weights_path,
             model_name=model_name)

if __name__ == "__main__":
    fire.Fire(train)
