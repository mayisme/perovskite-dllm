#!/bin/bash
# 快速验证脚本 - 用于测试配置是否正确
# 10 epochs, 小batch, 快速完成

set -e

PROJECT_DIR="/Users/xiaoyf/Documents/Python/dllm-perovskite"
cd "$PROJECT_DIR"

# 快速验证配置
CONFIGS=(
    "experiments/overnight_run/configs/lr_5e-5_warmup_fast.yaml"
    "experiments/overnight_run/configs/lr_2e-4_warmup_fast.yaml"
    "experiments/overnight_run/configs/lr_1e-4_linear_fast.yaml"
)

EXPERIMENT_NAMES=(
    "lr_5e-5_warmup_fast"
    "lr_2e-4_warmup_fast"
    "lr_1e-4_linear_fast"
)

LOG_DIR="experiments/overnight_run/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "快速验证实验开始 (10 epochs)"
echo "开始时间: $(date)"
echo "========================================"

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    EXP_NAME="${EXPERIMENT_NAMES[$i]}"
    
    echo ""
    echo "----------------------------------------"
    echo "实验 $((i+1))/3: $EXP_NAME"
    echo "配置文件: $CONFIG"
    echo "----------------------------------------"
    
    python main.py train --config "$CONFIG" --checkpoint-dir "experiments/overnight_run/checkpoints/${EXP_NAME}" 2>&1 | tee "$LOG_DIR/${EXP_NAME}.log"
    
    echo "✓ 实验 $EXP_NAME 完成"
done

echo ""
echo "========================================"
echo "快速验证完成!"
echo "结束时间: $(date)"
echo "========================================"
echo ""
echo "如果验证通过，可以运行完整实验:"
echo "  bash experiments/overnight_run/run_lr_experiments.sh"
