import json
import os
import sys
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

from collections import defaultdict
from matplotlib.lines import Line2D

from accelerate import Accelerator


RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from src.utils.utils import read_jsonl, write_jsonl, transformodel_name2model_path, load_model_tokenizer, get_model_wrapper
from sklearn.manifold import TSNE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1-8b-instruct', help='path to config file')
    parser.add_argument('--datasets', type=str, help="string of datasets", default="mathoai")
    parser.add_argument('--module', type=str, default="hidden", help="inject vector to which module, attn / mlp / hidden")
    parser.add_argument('--layers', type=str, default="0,1,2,3", help="layer to inject")
    parser.add_argument('--sample_num', type=int, default=100, help="sample number")
    parser.add_argument('--use_pred_res', action='store_true', help='use pred res')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser.parse_args()

def main():
    
    # ========== initialize ==========
    
    config = {
        "tok_pos": "last",
    }
    
    # get args
    args = get_args()
    accelerator = Accelerator()
    device = accelerator.device
    layers = args.layers.split(',')
    layers = [int(layer) for layer in layers]
    sample_num = args.sample_num
    
    module = args.module
    config['module'] = module
    model_name = args.model_name
    model_path = transformodel_name2model_path(model_name)
    model, tokenizer, model_config, MODEL_CONFIG = load_model_tokenizer(model_path, accelerator, output_hidden_states=True, load_in_8bit=False)
    model_wrapper = get_model_wrapper(model_name, model, tokenizer, model_config, accelerator)
    
    demon_path = "data/demon/longcot/long_short_form_thought.jsonl"
    
    instruction = "Answer the following question step by step and put the final answer in \\boxed{}:\n"
    
    # ========== get vector ==========
    
    pred_res_vector_list = []
    short_form_vector_list = []
    long_form_vector_list = []
    
    demon_data = read_jsonl(demon_path)
    if args.debug:
        demon_data = demon_data[:40]
    if args.use_pred_res:
        demon_data = demon_data[:100]

    demon_data = demon_data[:sample_num]

    for i, d in tqdm(enumerate(demon_data), desc="get vector"):
        problem = d['problem']
        short_form = d['short_form_thought']
        long_form = d['long_form_thought']
        
        # # ========== process short form demonstration ==========
        short_form_demon_list = [
            {"role": "user", "content": (instruction + problem).strip()},
            {"role": "assistant", "content": short_form}
        ]
        short_form_demon_str = tokenizer.apply_chat_template(short_form_demon_list, tokenize=False)    
        
        short_form_all_latent_dicts = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                short_form_demon_token = tokenizer(short_form_demon_str, return_tensors='pt').to(device)
                _ = model(**short_form_demon_token)
            short_form_all_latent_dicts.append(model_wrapper.latent_dict)
        
        short_form_context_vector_dict = model_wrapper.get_context_vector(short_form_all_latent_dicts, config)
        short_form_context_vector_dict = {key: value for key, value in short_form_context_vector_dict.items() if key in layers}
        short_form_vector_list.append(short_form_context_vector_dict)
        del short_form_all_latent_dicts
        
        # ========== process long form demonstration ==========
        long_form_demon_list = [
            {"role": "user", "content": (instruction + problem).strip()},
            {"role": "assistant", "content": long_form}
        ]
        long_form_demon_str = tokenizer.apply_chat_template(long_form_demon_list, tokenize=False)
        
        long_form_all_latent_dicts = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                long_form_demon_token = tokenizer(long_form_demon_str, return_tensors='pt').to(device)
                _ = model(**long_form_demon_token)
            long_form_all_latent_dicts.append(model_wrapper.latent_dict)
            
        long_form_context_vector_dict = model_wrapper.get_context_vector(long_form_all_latent_dicts, config)
        long_form_context_vector_dict = {key: value for key, value in long_form_context_vector_dict.items() if key in layers}
        long_form_vector_list.append(long_form_context_vector_dict)
        del long_form_all_latent_dicts

    
    for layer in tqdm(layers, desc='Plot'):

        short_form_vec_layer_module_list = []
        long_form_vec_layer_module_list = []
        for i, (short_form_vec, long_form_vec) in enumerate(zip(short_form_vector_list, long_form_vector_list)):
            short_form_vec_layer_module_list.append(short_form_vec[layer][module].numpy())
            long_form_vec_layer_module_list.append(long_form_vec[layer][module].numpy())
        short_form_vec_layer_module_np = np.array(short_form_vec_layer_module_list)
        long_form_vec_layer_module_np = np.array(long_form_vec_layer_module_list)
        
        data = np.vstack([short_form_vec_layer_module_np, long_form_vec_layer_module_np])
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        visualiser = TSNE(n_components=2, random_state=42)
        data_tsne = visualiser.fit_transform(data_scaled)

        data_tsne_list = data_tsne.tolist()

        output_file_path = f'exp-final/analysis/thinking_system_re/{model_name}'
        os.makedirs(output_file_path, exist_ok=True)
        with open(f"{output_file_path}/{layer}.json", 'w') as json_file:
            json.dump(data_tsne_list, json_file)
        
        # 将数据按组分开
        n_short = len(short_form_vec_layer_module_np)  # 假设短表单的数量等于short_form_vec_layer_module_np的长度
        n_long = len(long_form_vec_layer_module_np)    # 假设长表单的数量等于long_form_vec_layer_module_np的长度

        short_x, short_y = data_tsne[:n_short, 0], data_tsne[:n_short, 1]
        long_x, long_y = data_tsne[n_short:, 0], data_tsne[n_short:, 1]

        # 创建二维网格
        x_min, x_max = data_tsne[:, 0].min() - 5, data_tsne[:, 0].max() + 5
        y_min, y_max = data_tsne[:, 1].min() - 5, data_tsne[:, 1].max() + 5

        # 创建更高分辨率的网格（增加分辨率）
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

        # 计算每个网格点的密度（例如使用短表单和长表单的坐标）
        # 计算短表单的密度
        kde_short = gaussian_kde([short_x, short_y])
        zz_short = kde_short(np.vstack([xx.ravel(), yy.ravel()]))
        zz_short = zz_short.reshape(xx.shape)

        # 计算长表单的密度
        kde_long = gaussian_kde([long_x, long_y])
        zz_long = kde_long(np.vstack([xx.ravel(), yy.ravel()]))
        zz_long = zz_long.reshape(xx.shape)

        # 绘制等高线图
        plt.figure(figsize=(8, 6))

        # 绘制短表单的等高线图
        contour_short = plt.contour(xx, yy, zz_short, levels=20, cmap='Blues', alpha=0.9)

        # 绘制长表单的等高线图
        contour_long = plt.contour(xx, yy, zz_long, levels=20, cmap='Reds', alpha=0.9)

        short_color = contour_short.cmap(0.5)  # 中间的蓝色
        long_color = contour_long.cmap(0.5)    # 中间的橙色

        # 创建图例的手动图标
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=short_color, markersize=10, label='Short Form'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=long_color, markersize=10, label='Long Form')
        ]

        # 为每个图例添加标签
        # plt.legend(handles=legend_elements)

        # Remove ticks, x-axis, and y-axis
        plt.xticks([])
        plt.yticks([])

        # Set title
        plt.title(f"Layer {layer+1}", fontsize=30)

        # 保存图片到指定目录
        pic_save_dir = f"pic/analysis/global_vecsl_tsne/{model_name}/"
        os.makedirs(pic_save_dir, exist_ok=True)
        plt.savefig(f"{pic_save_dir}/{layer+1}_{module}.png", dpi=1000, bbox_inches='tight')
        plt.savefig(f"{pic_save_dir}/{layer+1}_{module}.pdf", dpi=1000, bbox_inches='tight')
        plt.close()
    
    
if __name__ == "__main__":
    set_seed(42)
    main()