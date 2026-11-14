#!/bin/bash

# Script para extração de features MediaPipe Multi-Core (AMD Threadripper)
# Ambiente recomendado: spamo_env_feat

ANNO_ROOT="./preprocess/Phoenix14T"
VIDEO_ROOT="phoenix_dl/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/"
SAVE_DIR="./datasets_features"
CACHE_DIR="./models"
MODEL_TYPE="auto"        # 'auto', 'holistic' ou 'separate'
NUM_WORKERS=16           # Seu Threadripper tem 64 threads lógicas (32 cores) - 28 processos é ideal!
WINDOW_SIZE=16
OVERLAP_SIZE=8
MODES=("dev" "test" "train")

echo "Iniciando extração de features MediaPipe paralela (${NUM_WORKERS} workers)..."
nohup python scripts/mediapipe_extract_feature.py \
    --anno_root "${ANNO_ROOT}" \
    --video_root "${VIDEO_ROOT}" \
    --save_dir "${SAVE_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --num_workers ${NUM_WORKERS} \
    --window_size ${WINDOW_SIZE} \
    --overlap_size ${OVERLAP_SIZE} \
    --mode "${MODES[@]}"

echo "Processamento concluído!"
