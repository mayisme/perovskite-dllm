#!/bin/bash
# 物理损失实验快速运行脚本
# 用于快速验证配置和代码，使用较少的epochs

set -e  # 遇到错误立即退出

PROJECT_DIR="/Users/xiaoyf/Documents/Python/dllm-perovskite"
cd "$PROJECT_DIR"

echo "=========================================="
echo "物理损失启用实验 - 快速测试"
echo "=========================================="
echo ""

# 实验配置
CONFIGS=("physics_0.05" "physics_0.1" "physics_0.2")
WEIGHTS=("0.05" "0.1" "0.2")
EPOCHS=5  # 快速测试只运行5个epochs

# 创建实验目录
EXPERIMENT_DIR="experiments/physics_loss_quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"

echo "实验目录: $EXPERIMENT_DIR"
echo "注意: 这是快速测试版本，只运行 $EPOCHS epochs"
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
    echo "开始训练 (快速模式: $EPOCHS epochs)..."
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
    
    echo ""
    echo "实验 $((i+1))/3 完成"
    echo ""
    
    # 短暂休息
    sleep 2
done

# 记录实验结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "=========================================="
echo "快速测试完成！"
echo "总耗时: ${MINUTES}分钟 ${SECONDS}秒"
echo "结果保存在: $EXPERIMENT_DIR"
echo "=========================================="
echo ""

# 生成实验摘要
echo "生成实验摘要..."
cat > "$EXPERIMENT_DIR/summary.txt" << EOF
物理损失启用实验摘要 (快速测试)
================================

实验时间: $(date)
总耗时: ${MINUTES}分钟 ${SECONDS}秒

实验配置:
---------
1. physics_0.05: 温和启用 (weight=0.05)
2. physics_0.1:  标准启用 (weight=0.1)
3. physics_0.2:  强约束 (weight=0.2)

训练参数:
---------
- Epochs: $EPOCHS (快速测试)
- Batch size: 16
- Learning rate: 1e-4
- 数据集: perovskites_20k_relaxed.h5

物理损失组件:
-----------
- Goldschmidt容忍因子: 0.1
- 配位数约束: 0.05
- 键长约束: 0.03
- Pauli排斥: 0.1

注意:
-----
这是快速测试版本，仅用于验证配置和代码。
完整训练请使用 run_physics_experiments.sh

结果文件:
---------
EOF

for CONFIG in "${CONFIGS[@]}"; do
    echo "- $CONFIG/:" >> "$EXPERIMENT_DIR/summary.txt"
    echo "  - best_model.pt" >> "$EXPERIMENT_DIR/summary.txt"
    echo "  - training.log" >> "$EXPERIMENT_DIR/summary.txt"
done

echo ""
echo "摘要文件: $EXPERIMENT_DIR/summary.txt"
echo ""
echo "下一步:"
echo "1. 查看训练日志: cat $EXPERIMENT_DIR/*/training.log"
echo "2. 如果测试通过，运行完整训练: ./run_physics_experiments.sh"
echo ""
