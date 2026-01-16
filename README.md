# SVSE

This work is currently submitted to ACL 2026, we built on top of the [VSE∞](https://github.com/woodfrog/vse_infty)
![Method](SVSE_Method.png)

The DPO is performed based on [ms-swift](https://github.com/modelscope/ms-swift), version 3.8.0.
DPO script:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type dpo \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --train_type lora \
    --dataset "path/to/dpo_dataset.jsonl" \
    --output_dir "output_dpo" \
    --torch_dtype bfloat16 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --max_length 4096 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 20
```
