#!/bin/bash
# 物理损失实验运行脚本
# 依次运行3个物理损失权重配置的训练实验

set -e  # 遇到错误立即退出

PROJECT_DIR="/Users/xiaoyf/Documents/Python/dllm-perovskite"
cd "$PROJECT_DIR"

echo "=========================================="
echo "物理损失启用实验 - 完整训练"
echo "=========================================="
echo ""

# 实验配置
CONFIGS=("physics_0.05" "physics_0.1" "physics_0.2")
WEIGHTS=("0.05" "0.1" "0.2")
EPOCHS=50

# 创建实验目录
EXPERIMENT_DIR="experiments/physics_loss_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"

echo "实验目录: $EXPERIMENT_DIR"
echo ""

# 记录实验开始时间
START_TIME=$(date +%s)

# 依次运行每个配置
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    WEIGHT="${WEIGHTS[$i]}"
    
    echo "=========================================="
    echo "实验 $((i+1))/3: 物理损失权重 = $WEIGHT"
    echo "配置文件: configs/${CONFIG}.yaml"
    echo "=========================================="
    echo ""
    
    # 创建实验子目录
    EXP_SUBDIR="$EXPERIMENT_DIR/${CONFIG}"
    mkdir -p "$EXP_SUBDIR"
    
    # 运行训练
    echo "开始训练..."
    python train.py \
        --config "configs/${CONFIG}.yaml" \
        --output_dir "$EXP_SUBDIR" \
        --epochs $EPOCHS \
        2>&1 | tee "$EXP_SUBDIR/training.log"
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "✓ 训练完成"
    else
        echo "✗ 训练失败，跳过后续实验"
        exit 1
    fi
    
    # 运行评估
    echo ""
    echo "开始评估..."
    python evaluate_model.py \
        --checkpoint "$EXP_SUBDIR/best_model.pt" \
        --config "configs/${CONFIG}.yaml" \
        --output_dir "$EXP_SUBDIR/evaluation" \
        2>&1 | tee "$EXP_SUBDIR/evaluation.log"
    
    if [ $? -eq 0 ]; then
        echo "✓ 评估完成"
    else
        echo "✗ 评估失败"
    fi
    
    echo ""
    echo "实验 $((i+1))/3 完成"
    echo ""
    
    # 短暂休息，避免系统过载
    sleep 5
done

# 记录实验结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "=========================================="
echo "所有实验完成！"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟"
echo "结果保存在: $EXPERIMENT_DIR"
echo "=========================================="
echo ""

# 生成实验摘要
echo "生成实验摘要..."
cat > "$EXPERIMENT_DIR/summary.txt" << EOF
物理损失启用实验摘要
==================

实验时间: $(date)
总耗时: ${HOURS}小时 ${MINUTES}分钟

实验配置:
---------
1. physics_0.05: 温和启用 (weight=0.05)
2. physics_0.1:  标准启用 (weight=0.1)
3. physics_0.2:  强约束 (weight=0.2)

训练参数:
---------
- Epochs: $EPOCHS
- Batch size: 16
- Learning rate: 1e-4
- 数据集: perovskites_20k_relaxed.h5

物理损失组件:
-----------
- Goldschmidt容忍因子: 0.1
- 配位数约束: 0.05
- 键长约束: 0.03
- Pauli排斥: 0.1

结果文件:
---------
EOF

for CONFIG in "${CONFIGS[@]}"; do
    echo "- $CONFIG/:" >> "$EXPERIMENT_DIR/summary.txt"
    echo "  - best_model.pt" >> "$EXPERIMENT_DIR/summary.txt"
    echo "  - training.log" >> "$EXPERIMENT_DIR/summary.txt"
    echo "  - evaluation/" >> "$EXPERIMENT_DIR/summary.txt"
done

echo ""
echo "摘要文件: $EXPERIMENT_DIR/summary.txt"
echo ""
echo "下一步:"
echo "1. 查看训练日志: cat $EXPERIMENT_DIR/*/training.log"
echo "2. 查看评估结果: cat $EXPERIMENT_DIR/*/evaluation.log"
echo "3. 对比不同权重的效果"
echo ""
