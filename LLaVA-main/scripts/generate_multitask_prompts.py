#原来没有的文件
import json
import random
import os

# 输入和输出文件
input_file = r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset/train.json"
output_file = r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset/train_with_prompts.json"

# 多任务提示模板
prompt_templates = [
    "请分析这幅画像，提供详细描述和分类信息。",
    "这幅画像属于什么类型？请提供详细描述和分类。",
    "请对这幅画像进行全面分析，包括描述和分类。",
    "这幅画像展示了什么内容？请详细描述并分类。",
    "请识别这幅画像的类型，并提供详细描述。",
    "分析这幅画像的特点，包括详细描述和分类。",
    "这是什么类型的画像？请提供详细描述。"
]

# 读取原始数据
with open(input_file, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 为每个样本随机选择一个提示模板
enhanced_dataset = []
for item in dataset:
    # 保留原始回答
    original_response = item["conversations"][1]["value"]

    # 随机选择一个提示模板
    prompt = random.choice(prompt_templates)

    # 更新提示
    new_item = item.copy()
    new_item["conversations"][0]["value"] = prompt

    enhanced_dataset.append(new_item)

# 保存增强后的数据集
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(enhanced_dataset, f, ensure_ascii=False, indent=2)

print(f"已生成多样化提示的数据集，包含{len(enhanced_dataset)}个样本")
