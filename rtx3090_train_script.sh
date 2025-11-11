#!/bin/bash
# RTX 3090 Optimized Training Script for MedSegDiff

# GPU optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Paths
DATA_DIR="/home/admin1/Documents/data/training"
OUT_DIR="./output_rtx3090_$(date +%Y%m%d_%H%M%S)"

# Training params optimized for RTX 3090 24GB
IMAGE_SIZE=256
BATCH_SIZE=16  # Optimal for 24GB VRAM
NUM_CHANNELS=128
MAX_STEPS=150000

echo "========================================="
echo "RTX 3090 Training Configuration"
echo "========================================="
echo "GPU: RTX 3090 (24GB VRAM)"
echo "Batch Size: $BATCH_SIZE (optimized for 24GB)"
echo "Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Max Steps: $MAX_STEPS"
echo "Expected time: ~50-75 hours (2-3 days) for 150K steps"
echo "Speed: ~2,000-3,000 steps/hour"
echo "Checkpoints saved every 2000 steps"
echo "========================================="
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Start training with optimized settings
python scripts/segmentation_train.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --image_size "$IMAGE_SIZE" \
  --num_channels "$NUM_CHANNELS" \
  --class_cond False \
  --num_res_blocks 2 \
  --num_heads 1 \
  --learn_sigma True \
  --use_scale_shift_norm False \
  --attention_resolutions 16 \
  --diffusion_steps 1000 \
  --noise_schedule linear \
  --rescale_learned_sigmas False \
  --rescale_timesteps False \
  --lr 1e-4 \
  --batch_size "$BATCH_SIZE" \
  --log_interval 50 \
  --save_interval 2000 \
  --lr_anneal_steps "$MAX_STEPS" \
  2>&1 | tee "$OUT_DIR/training.log"

echo ""
echo "========================================="
echo "Training completed at $(date)"
echo "========================================="
