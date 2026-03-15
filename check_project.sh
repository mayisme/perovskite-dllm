#!/bin/bash

echo "======================================"
echo "项目完整性检查"
echo "======================================"
echo ""

# 检查目录结构
echo "检查目录结构..."
dirs=("data" "models" "configs" "tests")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "✗ $dir/ (缺失)"
    fi
done
echo ""

# 检查核心文件
echo "检查核心文件..."
files=(
    "data/ionic_radii.py"
    "data/filter.py"
    "data/preprocess.py"
    "data/dataset.py"
    "models/diffusion.py"
    "models/egnn.py"
    "models/physics_loss.py"
    "train.py"
    "generate.py"
    "validate.py"
    "visualize.py"
    "main.py"
    "configs/base.yaml"
    "tests/test_modules.py"
    "example_workflow.py"
    "README.md"
    "requirements.txt"
    "PROJECT_SUMMARY.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (缺失)"
    fi
done
echo ""

# 统计代码行数
echo "代码统计..."
echo "Python文件数: $(find . -name "*.py" -not -path "./.kiro/*" | wc -l)"
echo "总代码行数: $(find . -name "*.py" -not -path "./.kiro/*" -exec wc -l {} + | tail -1 | awk '{print $1}')"
echo ""

# 运行测试
echo "运行测试..."
python tests/test_modules.py 2>&1 | grep -E "(✓|✗|passed|failed)"
echo ""

echo "======================================"
echo "检查完成！"
echo "======================================"
