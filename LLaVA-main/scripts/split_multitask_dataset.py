#原来没有的文件
import json
import random
import os

# 读取原始数据集
dataset_path = r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset/dataset.jsonll"
output_dir = r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset"

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 设置随机种子以确保可重复性
random.seed(42)
# 打乱数据集
random.shuffle(dataset)

# 分割比例（80%训练，20%验证）
train_ratio = 0.8
val_ratio = 0.2

train_size = int(len(dataset) * train_ratio)

train_data = dataset[:train_size]
val_data = dataset[train_size:]

# 保存训练集
with open(os.path.join(output_dir, "train.json"), 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

# 保存验证集
with open(os.path.join(output_dir, "val.json"), 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"数据集分割完成：")
print(f"- 训练集: {len(train_data)}个样本")
print(f"- 验证集: {len(val_data)}个样本")
