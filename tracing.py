import argparse
import warnings
warnings.filterwarnings("ignore")

import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import BitsAndBytesConfig
from utils import *

_method_configs = {
    'TracIn': {
        'feature_extractor': lambda p1, p2, p3, p4, p5, p6: get_gradient_vector(p1, p2, p3, p4),
        'metric': 'dot'
    },
    'TracInLN': {
        'feature_extractor': lambda p1, p2, p3, p4, p5, p6: TracInLN(p1, p2, p3, p4),
        'metric': 'dot'
    },
    'RapidIn': {
        'feature_extractor': lambda p1, p2, p3, p4, p5, p6: RapidIn(p1, p2, p3, p4),
        'metric': 'dot'
    },
    'RepT': {
        'feature_extractor': lambda p1, p2, p3, p4, p5, p6: instance_RepT(p1, p2, p3, p4, p5),
        'metric': 'cosine'
    },
    'LESS': {
        'feature_extractor': lambda p1, p2, p3, p4, p5, p6: LESS(p1, p2, p3, p4, p6),
        'metric': 'cosine'
    }
}

def tracing_undesirable_behaviors_offline(model, tokenizer, source_data, eval_data, cache, method, topk):
    score_path = 'cache/' + args.model + '/' + args.lora + '_' + method + '.npy'
    if not os.path.exists(score_path):
        if not os.path.exists('cache/' + args.model):
            os.mkdir('cache/' + args.model)
        print('Caching...')
        if method in ['DataInf', 'LiSSA', 'TEST']:
            source, eval = collect_gradient(model, tokenizer, source_data, eval_data)
            score = influence_function(source, eval, hvp_cal=method)
        if cache:
            np.save(score_path, score)
    else: score_path = np.load(score_path)
    print(method + ' Computing...')
    compute_metrics_offline(source_data, eval_data, score, topk)

def tracing_undesirable_behaviors(model, tokenizer, source_data, eval_data, cache, method, topk, layer, optimizer_state):
    source, eval = [], []
    tr_path = 'cache/' + args.model + '/' + args.lora + '_' + method + '_tr.pkl'
    val_path = 'cache/' + args.model + '/' + args.lora + '_' + method + '_val.pkl'
    if not os.path.exists(tr_path):
        if not os.path.exists('cache/' + args.model):
            os.mkdir('cache/' + args.model)
        print('Caching...')
        start_time = time.time()
        for idx in tqdm(range(len(source_data['prompts']))):
            source.append(_method_configs[method]['feature_extractor'](model, tokenizer, source_data['prompts'][idx], source_data['response'][idx], layer, optimizer_state)) 
        for idx in tqdm(range(len(eval_data['prompts']))):
            eval.append(_method_configs[method]['feature_extractor'](model, tokenizer, eval_data['prompts'][idx], eval_data['expected_response'][idx], layer, None).T)
        if cache:
            with open(tr_path, 'wb') as f:
                pickle.dump(source, f)
            with open(val_path, 'wb') as f:
                pickle.dump(eval, f)
        print(f"caching time: {time.time() - start_time:.4f} second")
    else:
        print('Loading...')
        with open(tr_path, 'rb') as f:
            source = pickle.load(f)
        with open(val_path, 'rb') as f:
            eval = pickle.load(f)

    print(method + ' Computing...')
    start_time = time.time()
    compute_metrics_online(source_data, eval_data, source, eval, topk, _method_configs[method]['metric'])
    print(f"matching time: {time.time() - start_time:.4f} second")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tracing undesirable behaviors")
    parser.add_argument('--model', type=str, default='llama2-7b', help='model name')
    parser.add_argument('--load_in_4bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--lora', type=str, default='', help='lora adapter')
    parser.add_argument('--cache', action='store_true', default=False, help='whether to cache the feature vector')
    parser.add_argument('--method', type=str, default='', help='tracing method: [Random, TracIn, RapidIn, RepT, DataInf, LiSSA, BM25]')
    parser.add_argument('--layer', type=int, default=-1, help='target layer')
    parser.add_argument('--topk', type=str, default='10', help='topk: 10 50 100')
    args = parser.parse_args()

    model_name = get_model_name(args.model)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if args.load_in_4bit else None
    lora_adapter_path = "lora_adapter/" + args.model + '/' + args.lora
    model, tokenizer = load_model_and_tokenizer(model_name, quantization_config, lora_adapter_path)
    optimizer_state = torch.load(lora_adapter_path + '/optimizer.pt', map_location='cpu')['state']
    model.eval()

    # format: [prompts, response, label]
    source_data = load_from_disk('datasets/' + args.lora[:args.lora.find('_')])
    # format: [prompts, expected response, label]
    eval_data = pd.read_csv('datasets/validation/' + args.lora + '_' + args.model + '.csv')
    topk = [int(_) for _ in args.topk.split()]
    if args.method in ['DataInf', 'LiSSA']:
        tracing_undesirable_behaviors_offline(model, tokenizer, source_data, eval_data, args.cache, args.method, topk)
    elif args.method in _method_configs.keys():
        tracing_undesirable_behaviors(model, tokenizer, source_data, eval_data, args.cache, args.method, topk, args.layer, optimizer_state)
    else: raise Exception("tracing method: [TracIn, RapidIn, LESS, RepT, DataInf, LiSSA, TracInLN]")
