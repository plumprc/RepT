import re
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import average_precision_score
from trak.projectors import BasicProjector, ProjectionType

norm = lambda x: (x - x.min()) / (x.max() - x.min())
sim = {
    'dot': lambda a, b: np.dot(a, b),
    'cosine': lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
}

def get_model_name(model):
    if model == 'tinyllama':
        return 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    elif model == 'llama2-7b':
        return '/common/public/LLAMA2-HF/Llama-2-7b-chat-hf'
    elif model == 'llama2-13b':
        return '/common/public/LLAMA2-HF/Llama-2-13b-chat-hf'
    elif model == 'llama2-70b':
        return '/common/public/LLAMA2-HF/Llama-2-70b-chat-hf'
    elif model == 'mistral':
        return 'mistralai/Mistral-7B-Instruct-v0.3'
    elif model == 'llama3':
        return '/common/public/LLAMA3.1/Meta-Llama-3.1-8B-Instruct'
    elif model == 'qwen2':
        return 'Qwen/Qwen2.5-7B-Instruct'
    else: raise Exception("model name: [tinyllama, llama2-7b, llama2-13b, mistral, llama3, qwen2]")

def load_model_and_tokenizer(model_name, quantization_config=None, lora_path=''):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        # torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=True) if len(lora_path) != 0 else model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_tokenized_text(tokenizer, sample, device, max_length=512):
    if tokenizer.chat_template:
        message = [
            {"role": "user", "content": sample['prompts']},
            {"role": "assistant", "content": sample['response']}
        ]
        full_text = tokenizer.apply_chat_template(message, tokenize=False)
        full_tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        prompt_only_text = tokenizer.apply_chat_template([message[0]], tokenize=False, add_generation_prompt=True)
        prompt_length = len(tokenizer(prompt_only_text, truncation=True, max_length=max_length, padding=False)['input_ids'])
    else:
        full_text = '[INST] ' + sample['prompts'] + ' [/INST]' + sample['response'] + tokenizer.eos_token
        full_tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        prompt_length = len(tokenizer('[INST] ' + sample['prompts'] + ' [/INST]', truncation=True, max_length=max_length, padding=False)['input_ids'])
    
    input_ids = full_tokenized['input_ids']
    labels = list(input_ids)
    labels[:prompt_length] = [-100] * prompt_length
    return {
        "input_ids": torch.tensor([input_ids]).to(device),
        "labels": torch.tensor([labels]).to(device),
        "attention_mask": torch.tensor([full_tokenized['attention_mask']]).to(device)
    }

def get_tokenized_dataset(tokenizer, dataset, max_length=512):
    def tokenize(sample):
        if tokenizer.chat_template:
            message = [
                {"role": "user", "content": sample['prompts']},
                {"role": "assistant", "content": sample['response']}
            ]
            full_text = tokenizer.apply_chat_template(message, tokenize=False)
            full_tokenized = tokenizer(full_text, truncation=True, max_length=512, padding=False)
            prompt_only_text = tokenizer.apply_chat_template([message[0]], tokenize=False, add_generation_prompt=True)
            prompt_length = len(tokenizer(prompt_only_text, truncation=True, max_length=max_length, padding=False)['input_ids'])
        else:
            full_text = '[INST] ' + sample['prompts'] + ' [/INST]' + sample['response'] + tokenizer.eos_token
            full_tokenized = tokenizer(full_text, truncation=True, max_length=512, padding=False)
            prompt_length = len(tokenizer('[INST] ' + sample['prompts'] + ' [/INST]', truncation=True, max_length=max_length, padding=False)['input_ids'])
        
        input_ids = full_tokenized['input_ids']
        labels = list(input_ids)
        labels[:prompt_length] = [-100] * prompt_length
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": full_tokenized['attention_mask']
        }    
    return dataset.map(tokenize, remove_columns=list(dataset.features))

def get_representation(model, tokenizer, prompt, layer):
    model.eval()
    if not (1 <= layer <= model.config.num_hidden_layers or layer == -1):
        raise ValueError(f"Layer index must be between 1 and {model.config.num_hidden_layers}. Got {layer}.")
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    else: prompt = '[INST] ' + prompt + ' [/INST]'
    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs['hidden_states'][layer][:, -1, :].view(-1).cpu().numpy()

def get_gradient_vector(model, tokenizer, prompt, expected_response):
    model.eval()
    model.zero_grad()
    inputs = get_tokenized_text(tokenizer, {'prompts': prompt, 'response': expected_response}, device=model.device)
    outputs = model(**inputs)
    loss = outputs['loss']
    loss.backward()
    gradient_vector = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    return gradient_vector.detach().cpu().numpy()

def TracInLN(model, tokenizer, prompt, expected_response):
    gradient_vector = get_gradient_vector(model, tokenizer, prompt, expected_response)
    gradient_vector = torch.Tensor(gradient_vector.reshape(model.config.num_hidden_layers, -1))
    gradient_vector = F.layer_norm(gradient_vector, normalized_shape=[gradient_vector.shape[-1]])
    return gradient_vector.view(-1).numpy()

def prepare_optimizer_state(model, optimizer_state):
    names = [_ for _ in range(len(optimizer_state))]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    avg = avg.to(model.device)
    avg_sq = avg_sq.to(model.device)
    return avg, avg_sq

def random_projection(vector, proj_dim=8192, block_size=128, model_id=0):
    projector = BasicProjector(grad_dim=vector.shape[0], proj_dim=proj_dim, seed=42, proj_type=ProjectionType.rademacher, device=vector.device, block_size=block_size)
    projected_v = projector.project(vector.reshape(1, -1), model_id=model_id)
    return projected_v.view(-1)

def LESS(model, tokenizer, prompt, expected_response, optimizer_state=None):
    model.eval()
    model.zero_grad()
    inputs = get_tokenized_text(tokenizer, {'prompts': prompt, 'response': expected_response}, device=model.device)
    outputs = model(**inputs)
    loss = outputs['loss']
    loss.backward()
    beta1, beta2, eps = 0.9, 0.999, 1e-08
    gradient_vector = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
    if optimizer_state is not None:
        avg, avg_sq = prepare_optimizer_state(model, optimizer_state)
        updated_avg = beta1 * avg + (1 - beta1) * gradient_vector
        updated_avg_sq = beta2 * avg_sq + (1 - beta2) * gradient_vector ** 2
        gradient_vector = updated_avg / torch.sqrt(updated_avg_sq + eps)
    
    gradient_vector = random_projection(gradient_vector)
    return gradient_vector.cpu().numpy()

def get_representation_gradient(model, tokenizer, prompt, expected_response, layer):
    model.eval()
    model.zero_grad()
    captured_grads = []
    if not (1 <= layer <= model.config.num_hidden_layers or layer == -1):
        raise ValueError(f"Layer index must be between 1 and {model.config.num_hidden_layers}. Got {layer}.")
    inputs = get_tokenized_text(tokenizer, {'prompts': prompt, 'response': expected_response}, device=model.device)
    outputs = model(**inputs, output_hidden_states=True)
    prompt_len = (inputs['labels'].cpu().numpy() == -100).sum()
    last_hidden_state = outputs['hidden_states'][layer]
    last_hidden_state.register_hook(lambda grad: captured_grads.append(grad))
    loss = outputs['loss']
    loss.backward()
    # [0, 0, ..., 0, 0, r_response_1, ..., r_response_n, 0]
    return captured_grads[0][0][prompt_len - 1:-1].cpu().numpy()

def instance_RepT(model, tokenizer, prompt, expected_response, layer):
    H = get_representation(model, tokenizer, prompt, layer)
    g_H = get_representation_gradient(model, tokenizer, prompt, expected_response, layer)[0]
    return np.hstack((H, g_H))

def _get_divisors(n):
    divs = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divs.append(i)     
            
    return divs

def random_suffle(vector, num_shuffles=20):
    vec_len = vector.shape[0] * vector.shape[1]
    shuffled_v = vector.clone()
    divs = _get_divisors(vec_len)
    # random shuffle
    for _ in range(num_shuffles):
        x_row = random.choice(divs)
        mat = shuffled_v.reshape(x_row, vec_len // x_row)
        row_indices = torch.randperm(mat.shape[0], device=vector.device)
        shuffled_v = mat[row_indices, :]

        x_col = random.choice(divs)
        mat = shuffled_v.reshape(vec_len // x_col, x_col)
        col_indices = torch.randperm(mat.shape[1], device=vector.device)
        shuffled_v = mat[:, col_indices]

    return shuffled_v.flatten()

def RapidIn(model, tokenizer, prompt, expected_response):
    model.eval()
    model.zero_grad()
    inputs = get_tokenized_text(tokenizer, {'prompts': prompt, 'response': expected_response}, device=model.device)
    outputs = model(**inputs)
    loss = outputs['loss']
    loss.backward()
    gradient_vector = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
    gradient_vector = gradient_vector.reshape(model.config.num_hidden_layers, -1)
    gradient_vector = F.layer_norm(gradient_vector, normalized_shape=[gradient_vector.shape[-1]])
    gradient_vector = random_suffle(gradient_vector)
    gradient_vector = random_projection(gradient_vector, proj_dim=2**16, block_size=1024)

    return gradient_vector.cpu().numpy()

def collect_gradient(model, tokenizer, source_data, eval_data):
    model.eval()
    tr_grad_dict = {}
    for idx in tqdm(range(len(source_data['prompts']))):
        model.zero_grad()
        inputs = get_tokenized_text(tokenizer, {'prompts': source_data['prompts'][idx], 'response': source_data['response'][idx]}, device=model.device)
        outputs = model(**inputs)
        loss = outputs['loss']
        loss.backward()
        grad_dict = {}
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            else: pass
        tr_grad_dict[idx] = grad_dict
        del grad_dict
            
    val_grad_dict = {}
    for idx in tqdm(range(len(eval_data['prompts']))):
        model.zero_grad()
        inputs = get_tokenized_text(tokenizer, {'prompts': eval_data['prompts'][idx], 'response': eval_data['expected_response'][idx]}, device=model.device)
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        grad_dict = {}
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            else: pass
        val_grad_dict[idx] = grad_dict    
        del grad_dict

    return tr_grad_dict, val_grad_dict

def influence_function(tr_grad_dict, val_grad_dict, hvp_cal='DataInf', lambda_const_param=1e3, n_iteration=0, alpha_const=1.):
    hvp_dict = defaultdict(dict)
    IF_dict = defaultdict(dict)
    n_train = len(tr_grad_dict.keys())

    def calculate_lambda_const(tr_grad_dict, weight_name):
        S = torch.zeros(len(tr_grad_dict.keys()))
        for tr_id in tr_grad_dict:
            tmp_grad = tr_grad_dict[tr_id][weight_name]
            S[tr_id] = torch.mean(tmp_grad**2)

        return torch.mean(S) / lambda_const_param

    if hvp_cal == 'DataInf':
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(tr_grad_dict, weight_name)
                hvp = torch.zeros(val_grad_dict[val_id][weight_name].shape)
                for tr_id in tr_grad_dict:
                    tmp_grad = tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (val_grad_dict[val_id][weight_name] - C_tmp * tmp_grad) / (n_train * lambda_const)
                
                hvp_dict[val_id][weight_name] = hvp

    elif hvp_cal == 'LiSSA':
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(tr_grad_dict, weight_name)
                running_hvp = val_grad_dict[val_id][weight_name]
                for _ in range(n_iteration):
                    hvp_tmp = torch.zeros(val_grad_dict[val_id][weight_name].shape)
                    for tr_id in tr_grad_dict:
                        tmp_grad = tr_grad_dict[tr_id][weight_name]
                        hvp_tmp += (torch.sum(tmp_grad * running_hvp) * tmp_grad - lambda_const * running_hvp) / n_train
                    
                    running_hvp = val_grad_dict[val_id][weight_name] + running_hvp - alpha_const * hvp_tmp

                hvp_dict[val_id][weight_name] = running_hvp

    else: raise Exception("hvp calculation options: [DataInf, LiSSA]")

    for tr_id in tqdm(tr_grad_dict):
        for val_id in val_grad_dict:
            if_tmp_value = 0
            for weight_name in val_grad_dict[0]:
                if_tmp_value += torch.sum(hvp_dict[val_id][weight_name] * tr_grad_dict[tr_id][weight_name])

            IF_dict[tr_id][val_id] = -if_tmp_value

    IF_score = pd.DataFrame(IF_dict, dtype=float).to_numpy()
    for eval_idx in range(len(IF_score)):
        IF_score[eval_idx] = norm(np.nan_to_num(IF_score[eval_idx], nan=np.nanmin(IF_score[eval_idx])))

    return IF_score

def compute_metrics_online(source_data, eval_data, source_vector, eval_vector, topk, metric):
    topk = sorted(topk, reverse=True)
    y_scores, y_trues, precision = {k: [] for k in topk}, {k: [] for k in topk}, {k: [] for k in topk}
    for i, vector in enumerate(tqdm(eval_vector)):
        sim_score = np.array([sim[metric](vector, vector_s) for vector_s in source_vector])
        y_trues_map = np.array([int(eval_data['label'][i] == label) for label in source_data['label']])
        top_max_k_indices = np.argsort(sim_score)[-topk[0]:]
        for k in topk:
            top_k_indices = top_max_k_indices[-k:]
            precision[k].append(y_trues_map[top_k_indices].sum() / k)
            for j in top_k_indices:
                y_scores[k].append(sim_score[j])
                y_trues[k].append(y_trues_map[j])

    for k in sorted(topk):
        print('top{} auPRC: {:.3f} P: {:.3f}'.format(str(k), average_precision_score(y_trues[k], y_scores[k]), np.mean(precision[k])))

def compute_metrics_offline(source_data, eval_data, matrix, topk):
    topk = sorted(topk, reverse=True)
    y_scores, y_trues, precision = {k: [] for k in topk}, {k: [] for k in topk}, {k: [] for k in topk}
    for i, item_score in enumerate(tqdm(matrix)):
        y_trues_map = np.array([int(eval_data['label'][i] == label) for label in source_data['label']])
        top_max_k_indices = np.argsort(item_score)[-topk[0]:]
        for k in topk:
            top_k_indices = top_max_k_indices[-k:]
            precision[k].append(y_trues_map[top_k_indices].sum() / k)
            for j in top_k_indices:
                y_scores[k].append(item_score[j])
                y_trues[k].append(y_trues_map[j])

    for k in sorted(topk):
        print('top{} auPRC: {:.3f} P: {:.3f}'.format(str(k), average_precision_score(y_trues[k], y_scores[k]), np.mean(precision[k])))
