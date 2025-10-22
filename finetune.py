import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from utils import get_model_name, load_model_and_tokenizer, get_tokenized_dataset
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForSeq2Seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='llama2-7b', help='model name')
    parser.add_argument('--load_in_4bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--max_length', type=int, default=512, help='tokenizer padding max length')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--logging_step', type=int, default=500, help='logging step')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--lora_r', type=int, default=4, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha')
    args = parser.parse_args()

    model_name = get_model_name(args.model)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if args.load_in_4bit else None
    model, tokenizer = load_model_and_tokenizer(model_name, quantization_config)

    dataset = load_from_disk("datasets/" + args.dataset)
    dataset = get_tokenized_dataset(tokenizer, dataset, max_length=args.max_length)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest'
    )

    training_args = TrainingArguments(
        output_dir="lora_adapter",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir="logs",
        logging_steps=args.logging_step,
        save_steps=10,
        save_total_limit=1
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    resume_from_checkpoint = args.resume if len(args.resume) != 0 else None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    lora_path = "lora_adapter/" + args.model
    if not os.path.exists(lora_path):
        os.mkdir(lora_path)
    trainer.save_model(lora_path + '/' + args.dataset + '_' + str(args.epochs))
    torch.save(trainer.optimizer.state_dict(), lora_path + '/' + args.dataset + '_' + str(args.epochs) + '/' + 'optimizer.pt')
    