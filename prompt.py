
import torch
from zeroprompthg import TxData, ZeroPrompt
import gc  # 引入垃圾回收模块
import os
import re
import glob
import argparse

parser = argparse.ArgumentParser(description='zeroprompthg')
parser.add_argument(
    "--model",
    type=str
)
args = parser.parse_args()
model_path = '/data/zhaojingtong/PrimeKG/our/random/lr0.001_batch2048_epochs30_hidden512_splitrandom_time12_22_21_43_seed12/fintune_model.pth'
# 设置根目录路径
base_dir = "/data/zhaojingtong/PrimeKG/our"
TxData = TxData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all')

TxData.prepare_split(split='random', seed=22, no_kg=False)
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
# 创建 TxGNN 实例
zeroprompt = ZeroPrompt(data=TxData,
                    weight_bias_track=False,
                    device=device)

zeroprompt.model_initialize(
    n_hid=512,
    n_inp=512,
    n_out=512,
    proto=True,
    proto_num=3,
    sim_measure='all_nodes_profile',
    bert_measure='disease_name',
    agg_measure='rarity',
    num_walks=200,
    walk_mode='bit',
    path_length=2
)

zeroprompt.prompt(
    n_epoch=1,
    learning_rate=5e-4,
    train_print_per_n=1500,
    valid_per_n=1500,
    model_path=model_path,
    save_result_path='./'
)

