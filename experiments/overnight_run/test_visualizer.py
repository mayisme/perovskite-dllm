#!/usr/bin/env python3
"""
测试脚本：验证 visualize_results.py 的所有功能

运行方式:
    python test_visualizer.py
"""

import sys
from pathlib import Path

def test_imports():
    """测试依赖包导入"""
    print("Testing imports...")
    try:
        import matplotlib
        import seaborn
        import h5py
        import numpy
        print("✓ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_visualizer_creation():
    """测试可视化工具创建"""
    print("\nTesting visualizer creation...")
    try:
        from visualize_results import ResultVisualizer
        visualizer = ResultVisualizer(base_dir=".")
        print(f"✓ Visualizer created, output dir: {visualizer.output_dir}")
        return True
    except Exception as e:
        print(f"✗ Error creating visualizer: {e}")
        return False

def test_loss_curves():
    """测试损失曲线绘制"""
    print("\nTesting loss curves plotting...")
    try:
        from visualize_results import ResultVisualizer
        visualizer = ResultVisualizer(base_dir=".")
        
        # 检查输入文件
        if not Path("phase1_analysis_report.json").exists():
            print("⚠ phase1_analysis_report.json not found, skipping test")
            return True
        
        visualizer.plot_loss_curves()
        
        # 检查输出文件
        if Path("figures/loss_curves.png").exists():
            print("✓ Loss curves generated successfully")
            return True
        else:
            print("✗ Loss curves file not created")
            return False
    except Exception as e:
        print(f"✗ Error plotting loss curves: {e}")
        return False

def test_sample_distribution():
    """测试样本分布绘制"""
    print("\nTesting sample distribution plotting...")
    try:
        from visualize_results import ResultVisualizer
        visualizer = ResultVisualizer(base_dir=".")
        
        # 检查输入文件
        if not Path("generated_samples.h5").exists():
            print("⚠ generated_samples.h5 not found, skipping test")
            return True
        
        visualizer.plot_sample_distribution()
        
        # 检查输出文件
        if Path("figures/sample_distribution.png").exists():
            print("✓ Sample distribution generated successfully")
            return True
        else:
            print("✗ Sample distribution file not created")
            return False
    except Exception as e:
        print(f"✗ Error plotting sample distribution: {e}")
        return False

def test_ablation_with_placeholder():
    """测试消融实验绘制（使用占位数据）"""
    print("\nTesting ablation plotting with placeholder data...")
    try:
        from visualize_results import ResultVisualizer
        visualizer = ResultVisualizer(base_dir=".")
        
        visualizer.plot_ablation_results()
        
        # 检查输出文件
        if Path("figures/ablation_results.png").exists():
            print("✓ Ablation results generated successfully (with placeholder data)")
            return True
        else:
            print("✗ Ablation results file not created")
            return False
    except Exception as e:
        print(f"✗ Error plotting ablation results: {e}")
        return False

def test_baseline_with_placeholder():
    """测试基线对比绘制（使用占位数据）"""
    print("\nTesting baseline comparison with placeholder data...")
    try:
        from visualize_results import ResultVisualizer
        visualizer = ResultVisualizer(base_dir=".")
        
        visualizer.plot_baseline_comparison()
        
        # 检查输出文件
        if Path("figures/baseline_comparison.png").exists():
            print("✓ Baseline comparison generated successfully (with placeholder data)")
            return True
        else:
            print("✗ Baseline comparison file not created")
            return False
    except Exception as e:
        print(f"✗ Error plotting baseline comparison: {e}")
        return False

def main():
    """运行所有测试"""
    print("="*60)
    print("Running visualize_results.py tests")
    print("="*60)
    
    tests = [
        test_imports,
        test_visualizer_creation,
        test_loss_curves,
        test_sample_distribution,
        test_ablation_with_placeholder,
        test_baseline_with_placeholder
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
