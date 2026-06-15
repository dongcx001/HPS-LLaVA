import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    print("DEBUG: Starting eval_model")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"DEBUG: Loading model: {model_name} from {model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    print("DEBUG: Loading questions")
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    print(f"DEBUG: Opened answers file: {answers_file}")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line.get("text", "No description available")  # 确保获取 text 字段
        print(f"DEBUG: Loaded text for question {idx}: {qs}")  # 调试：检查加载的 text
        cur_prompt = f"Using the provided description: '{qs}' as a strict reference, you MUST respond ONLY in English. You MUST start your response with the exact phrase 'Classification: [type]' followed by a newline, where [type] is determined as follows: if the description and image depict animals being chased, use '[animal chase]'; if they involve performances like dancing, music, or acrobatics, use '[dance and music]'; if they depict people in formal, symbolic, or social roles (e.g., officials, nobility, or symbolic figures), use '[social life]'. After the classification, provide a detailed analysis that includes every single detail from the description, such as specific animals, actions, objects, positions (e.g., left, middle, right), cultural context, and visual elements, following the description order exactly. Ensure no detail is omitted. If the classification cannot be determined, use 'Classification: [unknown]'."
        #cur_prompt = f"Using the provided description: '{qs}' as a strict reference, you MUST respond in English. First, determine the scene type based on the description and image: if it depicts animals being chased, use 'Classification: [animal chase]'; if it involves performances like dancing, music, or acrobatics, use 'Classification: [dance and music]'; if it depicts people in formal, symbolic, or social roles (e.g., officials, nobility, or symbolic figures), use 'Classification: [social life]'. You MUST begin your response with the classification in the exact format 'Classification: [type]' (e.g., 'Classification: [animal chase]'). Then, provide a detailed analysis that includes all details from the description, such as cultural context, specific actions, and visual elements, following the description order exactly. Ensure every detail is explicitly mentioned. If the classification cannot be determined, use 'Classification: [unknown]'."
       # cur_prompt = f"Using the provided description: '{qs}' as a strict reference, you MUST respond in English. You MUST begin with a classification in the exact format 'Classification: [type]' (e.g., 'Classification: animal chase'), followed by a detailed analysis that includes all details from the description, such as cultural context, specific actions, and visual elements. Ensure every detail is explicitly mentioned and follows the description order. If the classification is missing, default to 'Classification: [unknown]'."
        #cur_prompt = f"Using the provided description: '{qs}' as a strict reference, you MUST begin with a classification in the exact format 'Classification: [type]' (e.g., 'Classification: animal chase'), followed by a detailed analysis that includes all details from the description, such as cultural context, specific actions, and visual elements. Ensure every detail is explicitly mentioned and follows the description order. If the classification is missing, default to 'Classification: [unknown]'."
        #cur_prompt = f"Using the provided description: '{qs}' as a strict reference, begin with a classification in the exact format 'Classification: [type]' (e.g., 'Classification: animal chase'), followed by a detailed analysis that includes all details from the description, such as cultural context, specific actions, and visual elements. Ensure every detail is explicitly mentioned and follows the description order."
        print(f"DEBUG: Processing question {idx}, prompt: {cur_prompt}")
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(f"DEBUG: Generated prompt: {prompt}")

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        print(f"DEBUG: Input IDs shape: {input_ids.shape}")

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        print(f"DEBUG: Image tensor shape: {image_tensor.shape}")

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True
                )
                print(f"DEBUG: Generated output_ids: {output_ids}")

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                print(f"DEBUG: Question {idx}, Output IDs: {output_ids}, Decoded Output: {outputs}")

                # 检查并添加 Classification 标签
                if not outputs.startswith("Classification: "):
                    outputs = "Classification: [unknown]\n" + outputs

                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": line.get("text", "No description available"),  # 确保写入正确的 text
                    "answer": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {}
                }, ensure_ascii=False) + "\n")
                ans_file.flush()
                print(f"DEBUG: Wrote result for question {idx}")
        except Exception as e:
            print(f"ERROR: Question {idx} failed with exception: {e}")
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": line.get("text", "No description available"),
                "answer": "",
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {"error": str(e)}
            }, ensure_ascii=False) + "\n")
            ans_file.flush()

    ans_file.close()
    print("DEBUG: Finished eval_model")
 #原来，后面新加的上面

 #原来，基本下面的内容没变
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    #原来没有下面这一行
    parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling during generation")  # 添加 do_sample 参数
    args = parser.parse_args()

    eval_model(args)
