import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_from_disk
from utils import get_model_name, load_model_and_tokenizer, get_tokenized_dataset
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForSeq2Seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='llama2-7b', help='model name')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--max_length', type=int, default=512, help='tokenizer padding max length')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--logging_step', type=int, default=500, help='logging step')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    args = parser.parse_args()

    model_name = get_model_name(args.model)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
    model, tokenizer = load_model_and_tokenizer(model_name, quantization_config, '')

    dataset = load_from_disk("datasets/" + args.dataset)
    dataset = get_tokenized_dataset(tokenizer, dataset, max_length=args.max_length)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest'
    )

    training_args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        # optim="paged_adamw_8bit",
        logging_dir="logs",
        logging_steps=args.logging_step,
        save_strategy="epoch",
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    save_path = "checkpoints/" + args.model
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    trainer.save_model(save_path + '/' + args.dataset + '_' + str(args.epochs))
    torch.save(trainer.optimizer.state_dict(), save_path + '/' + args.dataset + '_' + str(args.epochs) + '/' + 'optimizer.pt')
    