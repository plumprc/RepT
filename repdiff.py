import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from transformers import BitsAndBytesConfig
from utils import get_model_name, load_model_and_tokenizer, get_representation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tracing undesirable behaviors")
    parser.add_argument('--model', type=str, default='llama2-7b', help='model name')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--lora', type=str, default='', help='lora adapter')
    parser.add_argument('--p', type=str, default='', help='input prompt')
    args = parser.parse_args()

    model_name = get_model_name(args.model)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
    lora_adapter_path = "lora_adapter/" + args.model + '/' + args.lora if len(args.lora) != 0 else ''
    model, tokenizer = load_model_and_tokenizer(model_name, quantization_config, lora_adapter_path)
    model.eval()

    sim = []
    cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    for layer in range(model.config.num_hidden_layers - 1):
        sim.append(cos(
            get_representation(model, tokenizer, args.p, layer + 1),
            get_representation(model, tokenizer, args.p, layer + 2)
        ))
    print(np.argmin(sim[:-1]) + 2)
