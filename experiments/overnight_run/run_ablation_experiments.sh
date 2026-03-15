#!/bin/bash
# 消融实验批量运行脚本
# 用途：自动运行4个消融实验配置，对比不同架构的性能

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/Users/xiaoyf/Documents/Python/dllm-perovskite"
cd "$PROJECT_ROOT"

# 实验配置
CONFIGS=(
    "pure_egnn"
    "pure_transformer"
    "hybrid_current"
    "hybrid_reverse"
)

# 实验描述
declare -A DESCRIPTIONS=(
    ["pure_egnn"]="纯EGNN架构（5层，无Transformer）"
    ["pure_transformer"]="纯Transformer架构（4层，无EGNN）"
    ["hybrid_current"]="当前混合架构（3层EGNN → 2层Transformer）"
    ["hybrid_reverse"]="反向混合架构（2层Transformer → 3层EGNN）"
)

# 日志目录
LOG_DIR="experiments/overnight_run/logs"
mkdir -p "$LOG_DIR"

# 结果目录
RESULTS_DIR="experiments/overnight_run/results"
mkdir -p "$RESULTS_DIR"

# 开始时间
START_TIME=$(date +%s)
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}消融实验批量运行${NC}"
echo -e "${GREEN}开始时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 运行每个实验
for config in "${CONFIGS[@]}"; do
    echo -e "${YELLOW}----------------------------------------${NC}"
    echo -e "${YELLOW}实验: $config${NC}"
    echo -e "${YELLOW}描述: ${DESCRIPTIONS[$config]}${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    
    CONFIG_PATH="experiments/overnight_run/configs/${config}.yaml"
    LOG_FILE="$LOG_DIR/${config}_$(date +%Y%m%d_%H%M%S).log"
    RESULT_DIR="$RESULTS_DIR/$config"
    
    mkdir -p "$RESULT_DIR"
    
    echo "配置文件: $CONFIG_PATH"
    echo "日志文件: $LOG_FILE"
    echo "结果目录: $RESULT_DIR"
    echo ""
    
    # 运行训练
    echo -e "${GREEN}开始训练...${NC}"
    if python scripts/train.py \
        --config "$CONFIG_PATH" \
        --output_dir "$RESULT_DIR" \
        > "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✓ 训练完成${NC}"
    else
        echo -e "${RED}✗ 训练失败，查看日志: $LOG_FILE${NC}"
        continue
    fi
    
    # 运行生成
    echo -e "${GREEN}开始生成样本...${NC}"
    if python scripts/generate.py \
        --config "$CONFIG_PATH" \
        --checkpoint "$RESULT_DIR/best_model.pt" \
        --output_dir "$RESULT_DIR/generated" \
        >> "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✓ 生成完成${NC}"
    else
        echo -e "${RED}✗ 生成失败，查看日志: $LOG_FILE${NC}"
        continue
    fi
    
    # 运行评估
    echo -e "${GREEN}开始评估...${NC}"
    if python scripts/evaluate.py \
        --config "$CONFIG_PATH" \
        --checkpoint "$RESULT_DIR/best_model.pt" \
        --output_dir "$RESULT_DIR/evaluation" \
        >> "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✓ 评估完成${NC}"
    else
        echo -e "${RED}✗ 评估失败，查看日志: $LOG_FILE${NC}"
    fi
    
    echo ""
done

# 结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成！${NC}"
echo -e "${GREEN}总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo -e "${GREEN}结束时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 生成对比报告
echo -e "${YELLOW}生成对比报告...${NC}"
python scripts/compare_ablation_results.py \
    --results_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/ablation_comparison.md"

echo -e "${GREEN}对比报告已生成: $RESULTS_DIR/ablation_comparison.md${NC}"
