import gradio as gr
import json
import os
import torch
from llava.model import LlavaLlamaForCausalLM
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

# 加载模型和处理器
def load_model_and_processor():
    model_path = "/home/dongchenxi/shiyan/llava/LLaVA-main/llava-v1.5-7b/"
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, processor, tokenizer

# 处理输入并生成结果
def process_image_and_text(image, description):
    try:
        model, processor, tokenizer = load_model_and_processor()

        prompt = f"Using the provided description: '{description}' as a strict reference, you MUST respond ONLY in English. You MUST start your response with the exact phrase 'Classification: [type]' followed by a newline, where [type] is determined as follows: if the description and image depict animals being chased, use '[animal chase]'; if they involve performances like dancing, music, or acrobatics, use '[dance and music]'; if they depict people in formal, symbolic, or social roles (e.g., officials, nobility, or symbolic figures), use '[social life]'. After the classification, provide a detailed analysis that includes every single detail from the description, such as specific animals, actions, objects, positions (e.g., left, middle, right), cultural context, and visual elements, following the description order exactly. Ensure no detail is omitted. If the classification cannot be determined, use 'Classification: [unknown]'."

        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        model.eval()
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=512, do_sample=False)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_text

    except Exception as e:
        return f"Error processing image and text: {str(e)}"

# 加载现有结果用于展示
def load_existing_results():
    results = {}
    merge_file = "/home/dongchenxi/shiyan/llava/LLaVA-main/playground/data/eval/hanhuaxiang/answers/llava-v1.5-7b/merge.jsonl"
    if os.path.exists(merge_file):
        with open(merge_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                if isinstance(data, dict):
                    question_id = data.get("question_id", "unknown")
                    answer = data.get("answer", "No result available")
                    results[question_id] = answer
    return results

# Gradio 界面
with gr.Blocks(title="LLaVA Image Analysis") as demo:
    gr.Markdown("# LLaVA Image Analysis Interface")
    gr.Markdown("Upload an image and enter a description to analyze using the LLaVA model.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            text_input = gr.Textbox(lines=2, label="Enter Description", placeholder="e.g., 该图描绘了古代狩猎或动物追逐场景...")
            submit_btn = gr.Button("Analyze")
        with gr.Column():
            output_text = gr.Textbox(label="Analysis Result", lines=5)
            existing_results = gr.Dataset(components=[gr.Textbox(label="Question ID"), gr.Textbox(label="Result")],
                                       samples=[(qid, result) for qid, result in load_existing_results().items()],
                                       type="index",
                                       label="Existing Results")

    submit_btn.click(
        fn=process_image_and_text,
        inputs=[image_input, text_input],
        outputs=output_text
    )

    print("Starting Gradio interface...")
    try:
        demo.launch(server_port=7864, debug=True)
    except Exception as e:
        print(f"Failed to launch Gradio: {str(e)}")
    print("Gradio interface started successfully or failed with details above.")
