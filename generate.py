import math
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from utils import get_model_name, load_model_and_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='llama2-7b', help='model name')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--lora', type=str, default='', help='lora adapter')
    parser.add_argument('--p', type=str, default='', help='evaluation prompt')
    parser.add_argument('--dataset', type=str, default='', help='evaluation dataset')
    parser.add_argument('--max_length', type=int, default=128, help='generation max length')
    parser.add_argument('--seg', type=int, default=100, help='number of test prompts')
    args = parser.parse_args()

    model_name = get_model_name(args.model)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
    lora_adapter_path = "lora_adapter/" + args.model + '/' + args.lora if len(args.lora) != 0 else ''
    model, tokenizer = load_model_and_tokenizer(model_name, quantization_config, lora_adapter_path)
    model.eval()

    # instruction = "Largely Rewrite the following question: {prompt}. Not too long. Rewrite version: "
    eval_data = pd.read_csv('datasets/' + args.dataset + '.csv')['prompts'].to_list() if len(args.dataset) != 0 else [args.p]
    # eval_data = [instruction.format(prompt=p) for p in eval_data]
    if tokenizer.chat_template:
        eval_data = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in eval_data]
    else: eval_data = ['[INST] ' + p + ' [/INST]' for p in eval_data]

    generated_texts = []
    for i in tqdm(range(math.ceil(len(eval_data) / args.seg))):
        inputs = tokenizer(eval_data[i * args.seg:(i+1) * args.seg], padding=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_length, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        generated_text = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generated_texts =  generated_texts + generated_text

    response_file_name = args.model + '_' + args.lora if len(args.lora) != 0 else args.model + '_' + args.dataset
    pd.DataFrame({'response': generated_texts}).to_csv('test_results/' + response_file_name + '.csv', index_label=False)
