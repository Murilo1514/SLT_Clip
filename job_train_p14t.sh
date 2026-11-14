export WANDB_MODE=offline


# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1


nohup python main.py -c configs/finetune.yaml -e bleu 
# conda deactivate
