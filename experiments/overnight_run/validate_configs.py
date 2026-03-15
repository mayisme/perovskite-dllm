#!/usr/bin/env python3
"""
验证学习率实验配置文件的正确性
"""

import yaml
import sys
from pathlib import Path

def validate_config(config_path):
    """验证单个配置文件"""
    print(f"\n检查配置文件: {config_path.name}")
    print("-" * 50)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查必要的字段
        required_sections = ['data', 'model', 'diffusion', 'training']
        for section in required_sections:
            if section not in config:
                print(f"  ❌ 缺少必要的配置节: {section}")
                return False
            else:
                print(f"  ✓ 配置节 '{section}' 存在")
        
        # 检查训练配置
        training = config['training']
        lr = training.get('lr')
        scheduler = training.get('lr_scheduler')
        epochs = training.get('epochs')
        
        print(f"\n  训练配置:")
        print(f"    - 学习率: {lr}")
        print(f"    - 调度器: {scheduler}")
        print(f"    - Epochs: {epochs}")
        
        if 'warmup_steps' in training:
            print(f"    - Warmup steps: {training['warmup_steps']}")
        
        # 检查学习率范围
        if lr < 1e-6 or lr > 1e-2:
            print(f"  ⚠️  警告: 学习率 {lr} 可能不在合理范围内")
        
        # 检查调度器
        valid_schedulers = ['cosine', 'linear', 'exponential', 'step']
        if scheduler and scheduler not in valid_schedulers:
            print(f"  ⚠️  警告: 调度器 '{scheduler}' 可能不被支持")
        
        print(f"  ✓ 配置文件格式正确")
        return True
        
    except yaml.YAMLError as e:
        print(f"  ❌ YAML解析错误: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 验证失败: {e}")
        return False

def main():
    """主函数"""
    project_dir = Path("/Users/xiaoyf/Documents/Python/dllm-perovskite")
    config_dir = project_dir / "experiments/overnight_run/configs"
    
    # 学习率实验配置文件
    lr_configs = [
        "lr_5e-5_warmup.yaml",
        "lr_2e-4_warmup.yaml",
        "lr_1e-4_linear.yaml",
        "lr_5e-5_warmup_fast.yaml",
        "lr_2e-4_warmup_fast.yaml",
        "lr_1e-4_linear_fast.yaml",
    ]
    
    print("=" * 60)
    print("学习率实验配置文件验证")
    print("=" * 60)
    
    all_valid = True
    for config_name in lr_configs:
        config_path = config_dir / config_name
        if not config_path.exists():
            print(f"\n❌ 配置文件不存在: {config_name}")
            all_valid = False
            continue
        
        if not validate_config(config_path):
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ 所有配置文件验证通过!")
        print("\n下一步:")
        print("  1. 快速验证: bash experiments/overnight_run/run_lr_experiments_fast.sh")
        print("  2. 完整训练: bash experiments/overnight_run/run_lr_experiments.sh")
    else:
        print("❌ 部分配置文件验证失败，请检查错误信息")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
