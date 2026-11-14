python scripts/vjepa2_extract_feature.py \
    --anno_root  ./preprocess/Phoenix14T \
    --video_root ./phoenix_dl/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ \
    --save_dir ./datasets_features \
    --model_name  facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --cache_dir  ./cache/ \
    --overlap_size   8 \
    --batch_size  4 \
    --device  cpu \
    --frames_per_clip  16 




