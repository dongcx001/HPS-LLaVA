#原来没有的文件
import os
import json
import glob
from PIL import Image

# 图像目录
image_dir = r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset/images/"
# 输出JSON文件
output_file = r"/home/dongchenxi/shiyan/llava/LLaVA-main/data/multitask_dataset/dataset.jsonl"

# 确保目录存在
os.makedirs(image_dir, exist_ok=True)

dataset = []
image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))

for idx, image_file in enumerate(image_files):
    # 使用相对路径
    rel_path = os.path.relpath(image_file, r"d:\pytorch-fuxian\LLaVA-main\data\multitask_dataset")
    rel_path = rel_path.replace("\\", "/")  # 确保路径格式一致

    # 创建多任务数据项
    item = {
        "id": str(idx),
        "image": rel_path,
        "conversations": [
            {
                "from": "human",
                "value": "请分析这幅画像，提供详细描述和分类信息。"
            },
            {
                "from": "gpt",
                "value": "描述：[这里填写画像描述]\n分类：[画像类型]"
            }
        ]
    }
    dataset.append(item)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"已生成多任务数据集模板，包含{len(dataset)}个样本，保存至{output_file}")
print(f"请手动编辑{output_file}文件，为每个图像提供准确的多任务标注信息")
