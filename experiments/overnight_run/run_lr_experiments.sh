#!/bin/bash
# 学习率调优实验批量训练脚本
# 用途: 依次运行3个学习率配置的训练实验
# 注意: CPU训练较慢，建议使用快速验证版本或在GPU上运行

set -e  # 遇到错误立即退出

# 工作目录
PROJECT_DIR="/Users/xiaoyf/Documents/Python/dllm-perovskite"
cd "$PROJECT_DIR"

# 实验配置
CONFIGS=(
    "experiments/overnight_run/configs/lr_5e-5_warmup.yaml"
    "experiments/overnight_run/configs/lr_2e-4_warmup.yaml"
    "experiments/overnight_run/configs/lr_1e-4_linear.yaml"
)

EXPERIMENT_NAMES=(
    "lr_5e-5_warmup"
    "lr_2e-4_warmup"
    "lr_1e-4_linear"
)

# 日志目录
LOG_DIR="experiments/overnight_run/logs"
mkdir -p "$LOG_DIR"

# 记录开始时间
START_TIME=$(date +%s)
echo "========================================" | tee -a "$LOG_DIR/batch_run.log"
echo "学习率调优实验批量训练开始" | tee -a "$LOG_DIR/batch_run.log"
echo "开始时间: $(date)" | tee -a "$LOG_DIR/batch_run.log"
echo "========================================" | tee -a "$LOG_DIR/batch_run.log"

# 依次运行每个实验
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    EXP_NAME="${EXPERIMENT_NAMES[$i]}"
    
    echo "" | tee -a "$LOG_DIR/batch_run.log"
    echo "----------------------------------------" | tee -a "$LOG_DIR/batch_run.log"
    echo "实验 $((i+1))/3: $EXP_NAME" | tee -a "$LOG_DIR/batch_run.log"
    echo "配置文件: $CONFIG" | tee -a "$LOG_DIR/batch_run.log"
    echo "开始时间: $(date)" | tee -a "$LOG_DIR/batch_run.log"
    echo "----------------------------------------" | tee -a "$LOG_DIR/batch_run.log"
    
    # 运行训练
    EXP_START=$(date +%s)
    python train.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/${EXP_NAME}.log"
    EXP_END=$(date +%s)
    EXP_DURATION=$((EXP_END - EXP_START))
    
    echo "实验 $EXP_NAME 完成" | tee -a "$LOG_DIR/batch_run.log"
    echo "耗时: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m $((EXP_DURATION % 60))s" | tee -a "$LOG_DIR/batch_run.log"
    
    # 检查checkpoint是否生成
    if [ -d "checkpoints/${EXP_NAME}" ]; then
        echo "✓ Checkpoint已保存到 checkpoints/${EXP_NAME}" | tee -a "$LOG_DIR/batch_run.log"
    else
        echo "⚠ 警告: 未找到checkpoint目录" | tee -a "$LOG_DIR/batch_run.log"
    fi
done

# 记录结束时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_DIR/batch_run.log"
echo "========================================" | tee -a "$LOG_DIR/batch_run.log"
echo "所有实验完成!" | tee -a "$LOG_DIR/batch_run.log"
echo "结束时间: $(date)" | tee -a "$LOG_DIR/batch_run.log"
echo "总耗时: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s" | tee -a "$LOG_DIR/batch_run.log"
echo "========================================" | tee -a "$LOG_DIR/batch_run.log"

# 生成结果摘要
echo "" | tee -a "$LOG_DIR/batch_run.log"
echo "实验结果摘要:" | tee -a "$LOG_DIR/batch_run.log"
echo "- 配置文件: experiments/overnight_run/configs/" | tee -a "$LOG_DIR/batch_run.log"
echo "- 训练日志: experiments/overnight_run/logs/" | tee -a "$LOG_DIR/batch_run.log"
echo "- Checkpoints: checkpoints/" | tee -a "$LOG_DIR/batch_run.log"
echo "" | tee -a "$LOG_DIR/batch_run.log"
echo "下一步:" | tee -a "$LOG_DIR/batch_run.log"
echo "1. 使用 analyze_results.py 分析训练曲线" | tee -a "$LOG_DIR/batch_run.log"
echo "2. 使用 evaluate_model.py 评估最佳模型" | tee -a "$LOG_DIR/batch_run.log"
echo "3. 根据结果选择最优学习率配置" | tee -a "$LOG_DIR/batch_run.log"
