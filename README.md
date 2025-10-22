# Tracing Undesirable LLM Behaviors
This is an official implementation of [Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing](https://arxiv.org/pdf/2510.02334)

## Get Started
1. Run `pip install -r requirements.txt`
2. Run `python finetune.py --dataset <dataset_name> --model <model_name>` to train the model
3. Run `python generate.py --model <model_name> --lora <lora_adapter> --dataset <test_data>` to check whether any undesirable behaviors occur on the test set, and then put them in `./datasets/validation/`
4. Run `python tracing.py --model <model_name> --lora <lora_adapter> --method <tracing_method> --topk <topk>` to trace the detected undesirable behaviors back to the corresponding training samples

For full-funetuning, run `python full_funetune.py --dataset <dataset_name> --model <model_name>`. After training, modify the model loading logic in `generate.py` and `tracing.py` to load your fully fine-tuned model. Refer to these code files for implementation details.

## Citation

If you find this repo useful, please cite our paper. 

```
@article{li2025did,
  title={Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing},
  author={Li, Zhe and Zhao, Wei and Li, Yige and Sun, Jun},
  journal={arXiv preprint arXiv:2510.02334},
  year={2025}
}
```

## Contact

If you have any questions or want to discuss some details, please contact zheli@smu.edu.sg.

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/ykwon0407/DataInf

https://github.com/princeton-nlp/LESS
