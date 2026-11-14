export WANDB_MODE=offline


# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1


# nohup python main.py -c configs/finetune.yaml -e bleu --train False --test True --ckpt logs/2026-11-04T19-45-22_finetune/checkpoints/epoch=00047-step=0007104-bleu4=19.16.ckpt
# conda deactivate

nohup python main.py -c configs/finetune.yaml -e bleu --train False --test True --ckpt logs/2026-11-08T17-58-12_finetune/checkpoints/epoch=00029-step=0013320-bleu4=15.48.ckpt