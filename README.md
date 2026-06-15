---
license: llama2
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- vision-language-model
- cultural-heritage
- han-dynasty
- image-inpainting
- chinese-culture
---

# HPS-LLaVA：面向残损汉画像石的视觉语言助手

## 模型详情

**模型类型**：多模态视觉语言模型（图像+文本 → 文本）

**基座模型**：[LLaVA-1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b)

**论文**：[待补充链接]

## 核心创新

| 创新点 | 描述 |
|--------|------|
| 🔧 **分数阶微分增强修复** | 在 RePaint 扩散模型中引入分数阶梯度引导，专门恢复石刻高频纹理细节 |
| 🎓 **双阶段并行式学习** | 阶段一：文化概念对齐（对比学习）；阶段二：端到端指令微调 |
| 📖 **汉画像石指令数据集** | 首个覆盖六大文化主题的多轮对话数据集（2,064图 / 8,064指令对） |

## 模型架构

残损汉画像石图像
    ↓
【分数阶微分增强修复模块】
    ↓
修复后图像
    ↓
【视觉编码器 CLIP-ViT-L/14】 (冻结)
    ↓
【投影层 MLP】 (阶段一训练)
    ↓
【大语言模型 Vicuna-13B】 (阶段二微调)
    ↓
文化理解文本输出

# 用途

## 主要用途

- 残损汉画像石的文化解读
- 汉代文化符号识别（西王母、伏羲女娲、车马出行、羽人戏兽等）
- 多轮对话式文物问答

## 适用场景

- 文化遗产数字化保护
- 考古辅助研究
- 博物馆智能导览

## 训练数据

| 数据类型 | 来源 | 规模 |
|----------|------|------|
| 文化概念对齐 | 专家标注图文对 | 2,048对 |
| 指令微调 | 自建汉画像石数据集 | 8,064组多轮对话 |

**覆盖题材**：社会生活、舞乐百戏、角抵驯兽、辟邪升仙、祥禽瑞兽、天文神话

**文化术语示例**：西王母、东王公、伏羲女娲交尾、四神云气、羽人戏兽、建鼓、九尾狐、三青鸟

## 使用方式

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+（推荐）

### 安装

```bash
pip install transformers torch accelerate pillow
```


## 推理代码

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained(
    "你的HuggingFace用户名/hps-llava",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("你的HuggingFace用户名/hps-llava")

image = Image.open("hanstone_image.jpg")
prompt = "请描述这幅汉画像石的内容"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```
## 局限性
模型性能依赖于所构建数据集的规模与质量

对罕见或学界有争议的汉画像石符号泛化能力有待验证

修复与理解模块的协同机制仍较为初步

## 未来工作
扩展数据集涵盖更边缘的文化意象

探索可迭代的修复-理解反馈机制

迁移至壁画、彩塑等其他文化遗产

## 引用
bibtex
@article{hpsllava2026,
  title={面向残损汉画像石的视觉语言助手},
  journal={（待补充）},
  year={2026}
}
## 致谢
本工作基于以下项目构建：

LLaVA - Visual Instruction Tuning

RePaint - Diffusion-based Inpainting

## 许可证
本项目继承 LLaVA 的许可证，基于 LLAMA 2 Community License 使用。
