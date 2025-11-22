This project fine-tunes the LLaVA-1.5-7B vision-language model on the VQA-RAD dataset using the `ms-swift` framework.

## Environment Setup
```
conda create --name ms-swift python=3.11
conda activate ms-swift
pip install -r requirements.txt
```

## Run zero-shot experiments for LLaVA
```bash
python zero-shot.py
```
This runs zero-shot evaluation with the base LLaVA model before fine-tuning.

## Fine-tune LLaVA
Download llava-1.5-7b-hf from Hugging Face (https://huggingface.co/llava-hf/llava-1.5-7b-hf) and place it in `./LLava_huggingface`

Run `preprocess.ipynb` to preprocess the VQA-RAD dataset into the ms-swift fine-tuning format; the processed data is saved in `./converted_vqa_rad`
```
preprocess.ipynb
```
#### Fine-tune LLaVA model
```
CUDA_VISIBLE_DEVICES=0 swift sft \
     --model_type llava1_5_hf \
     --model ./LLava_huggingface \
     --dataset ./converted_vqa_rad/train/train.jsonl \
     --val_dataset ./converted_vqa_rad/val/val.jsonl \
     --train_type lora \
     --output_dir output \
     --num_train_epochs 5 \
     --per_device_train_batch_size 4 \
     --per_device_eval_batch_size 4 \
     --learning_rate 1e-4 \
     --logging_steps 10 \
     --eval_steps 20 \
     --save_steps 50
```

## Merge trained LoRA weights with LLaVA model
```
swift export \
    --ckpt_dir output/checkpoint \
    --merge_lora true
```
The merged model can then be used for inference in place of the original LLaVA checkpoint.