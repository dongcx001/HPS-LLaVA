#原来没有的文件
import json
import os
import re
from PIL import Image


def validate_multitask_dataset(dataset_path, image_base_dir):
    try:
        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"数据集包含{len(dataset)}个样本")

        # 验证每个样本
        valid_samples = 0
        issues = []

        # 定义多任务输出的正则表达式模式
        pattern = r"描述：.*\n分类：.*"

        for idx, item in enumerate(dataset):
            # 检查必要字段
            if "id" not in item or "image" not in item or "conversations" not in item:
                issues.append(f"样本{idx}缺少必要字段")
                continue

            # 检查图像文件是否存在
            image_path = os.path.join(image_base_dir, item["image"])
            if not os.path.exists(image_path):
                issues.append(f"样本{idx}的图像文件不存在: {image_path}")
                continue

            # 检查对话格式
            if len(item["conversations"]) < 2:
                issues.append(f"样本{idx}的对话数量不足")
                continue

            if item["conversations"][0]["from"] != "human" or item["conversations"][1]["from"] != "gpt":
                issues.append(f"样本{idx}的对话角色顺序不正确")
                continue

            # 检查多任务输出格式
            response = item["conversations"][1]["value"]
            if not re.search(pattern, response):
                issues.append(f"样本{idx}的回答格式不符合多任务要求")
                continue

            # 尝试打开图像文件
            try:
                img = Image.open(image_path)
                img.close()
            except Exception as e:
                issues.append(f"样本{idx}的图像文件无法打开: {str(e)}")
                continue

            valid_samples += 1

        print(f"有效样本数: {valid_samples}/{len(dataset)}")
        if issues:
            print("发现以下问题:")
            for issue in issues[:10]:  # 只显示前10个问题
                print(f"- {issue}")
            if len(issues) > 10:
                print(f"... 还有{len(issues) - 10}个问题未显示")
        else:
            print("数据集验证通过，未发现问题")

    except Exception as e:
        print(f"验证过程出错: {str(e)}")


# 验证训练集
validate_multitask_dataset(
    r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset/train.json",
    r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset"
)

# 验证验证集
validate_multitask_dataset(
    r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset/val.json",
    r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset"
)
