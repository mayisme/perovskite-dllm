#!/usr/bin/env python3
"""
简化的ARIS风格模型分析脚本
分析训练好的模型的架构和参数
"""

import torch
import json
from pathlib import Path
from datetime import datetime
import argparse

from models.hybrid_model import HybridEGNNTransformer
from utils.config_utils import load_config


def analyze_model(checkpoint_path, config_path, output_dir='evaluation_results'):
    """分析模型架构和参数"""
    
    print("=" * 60)
    print("ARIS Model Analysis")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建模型
    print("\n[1/4] Creating model architecture...")
    model = HybridEGNNTransformer(
        hidden_dim=config['model']['hidden_dim'],
        n_egnn_layers=config['model']['n_egnn_layers'],
        n_transformer_layers=config['model']['n_transformer_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        cutoff=config['model'].get('cutoff_radius', 6.0)
    )
    
    # 加载权重
    print("[2/4] Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 分析参数
    print("[3/4] Analyzing parameters...")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 分析各组件
    component_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        component_params[name] = params
    
    # 分析EGNN vs Transformer
    egnn_params = sum(p.numel() for name, p in model.named_parameters() if 'egnn' in name)
    transformer_params = sum(p.numel() for name, p in model.named_parameters() if 'transformer' in name)
    embedding_params = sum(p.numel() for name, p in model.named_parameters() if 'emb' in name)
    output_params = sum(p.numel() for name, p in model.named_parameters() if 'out_' in name)
    
    results = {
        'metadata': {
            'checkpoint': str(checkpoint_path),
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'training_epoch': checkpoint.get('epoch', 'unknown'),
            'training_loss': checkpoint.get('loss', 'unknown')
        },
        'architecture': {
            'hidden_dim': config['model']['hidden_dim'],
            'n_egnn_layers': config['model']['n_egnn_layers'],
            'n_transformer_layers': config['model']['n_transformer_layers'],
            'num_heads': config['model']['num_heads'],
            'dropout': config['model']['dropout'],
            'cutoff_radius': config['model'].get('cutoff_radius', 6.0)
        },
        'parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'by_component': component_params,
            'by_type': {
                'egnn': egnn_params,
                'transformer': transformer_params,
                'embedding': embedding_params,
                'output': output_params
            },
            'ratios': {
                'egnn': egnn_params / total_params,
                'transformer': transformer_params / total_params,
                'embedding': embedding_params / total_params,
                'output': output_params / total_params
            }
        }
    }
    
    # 打印结果
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Model Architecture:")
    print(f"  - Hidden dimension: {results['architecture']['hidden_dim']}")
    print(f"  - EGNN layers: {results['architecture']['n_egnn_layers']}")
    print(f"  - Transformer layers: {results['architecture']['n_transformer_layers']}")
    print(f"  - Attention heads: {results['architecture']['num_heads']}")
    print(f"  - Dropout: {results['architecture']['dropout']}")
    print(f"  - Cutoff radius: {results['architecture']['cutoff_radius']} Å")
    
    print(f"\n🔢 Parameter Count:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    
    print(f"\n🧩 Component Breakdown:")
    for name, params in component_params.items():
        ratio = params / total_params
        print(f"  - {name}: {params:,} ({ratio:.1%})")
    
    print(f"\n🏗️  Architecture Distribution:")
    print(f"  - EGNN: {egnn_params:,} ({results['parameters']['ratios']['egnn']:.1%})")
    print(f"  - Transformer: {transformer_params:,} ({results['parameters']['ratios']['transformer']:.1%})")
    print(f"  - Embeddings: {embedding_params:,} ({results['parameters']['ratios']['embedding']:.1%})")
    print(f"  - Output layers: {output_params:,} ({results['parameters']['ratios']['output']:.1%})")
    
    print(f"\n📈 Training Info:")
    print(f"  - Epoch: {results['metadata']['training_epoch']}")
    print(f"  - Loss: {results['metadata']['training_loss']}")
    
    # 保存结果
    print("\n[4/4] Saving results...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_path / f'model_analysis_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 生成报告
    report_file = output_path / f'model_report_{timestamp}.md'
    generate_report(results, report_file)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Report saved to: {report_file}")
    
    return results


def generate_report(results, output_path):
    """生成Markdown报告"""
    
    report = f"""# Model Analysis Report

**Date**: {results['metadata']['timestamp']}
**Checkpoint**: {results['metadata']['checkpoint']}
**Training Epoch**: {results['metadata']['training_epoch']}
**Training Loss**: {results['metadata']['training_loss']}

## Architecture

| Component | Value |
|-----------|-------|
| Hidden Dimension | {results['architecture']['hidden_dim']} |
| EGNN Layers | {results['architecture']['n_egnn_layers']} |
| Transformer Layers | {results['architecture']['n_transformer_layers']} |
| Attention Heads | {results['architecture']['num_heads']} |
| Dropout | {results['architecture']['dropout']} |
| Cutoff Radius | {results['architecture']['cutoff_radius']} Å |

## Parameter Statistics

**Total Parameters**: {results['parameters']['total']:,}
**Trainable Parameters**: {results['parameters']['trainable']:,}

### Distribution by Type

| Type | Parameters | Percentage |
|------|------------|------------|
| EGNN | {results['parameters']['by_type']['egnn']:,} | {results['parameters']['ratios']['egnn']:.1%} |
| Transformer | {results['parameters']['by_type']['transformer']:,} | {results['parameters']['ratios']['transformer']:.1%} |
| Embeddings | {results['parameters']['by_type']['embedding']:,} | {results['parameters']['ratios']['embedding']:.1%} |
| Output Layers | {results['parameters']['by_type']['output']:,} | {results['parameters']['ratios']['output']:.1%} |

### Component Breakdown

"""
    
    for name, params in results['parameters']['by_component'].items():
        ratio = params / results['parameters']['total']
        report += f"- **{name}**: {params:,} ({ratio:.1%})\n"
    
    report += f"""

## Key Insights

1. **Balanced Architecture**: The model has a {results['parameters']['ratios']['egnn']:.1%} EGNN and {results['parameters']['ratios']['transformer']:.1%} Transformer split, showing a balanced hybrid design.

2. **Parameter Efficiency**: With {results['parameters']['total']:,} total parameters, the model is relatively compact for a hybrid architecture.

3. **Local-Global Balance**: 
   - EGNN layers ({results['architecture']['n_egnn_layers']}) capture local geometric features
   - Transformer layers ({results['architecture']['n_transformer_layers']}) model global dependencies

4. **Attention Mechanism**: {results['architecture']['num_heads']} attention heads provide multi-scale feature aggregation.

## Next Steps

### Immediate Actions
1. ✅ Model architecture analyzed
2. ⏳ Generate samples using `generate.py`
3. ⏳ Evaluate sample validity
4. ⏳ Assess diversity and novelty

### Optimization Opportunities
1. **Ablation Studies**:
   - Pure EGNN (no Transformer)
   - Pure Transformer (no EGNN)
   - Different layer ratios

2. **Hyperparameter Tuning**:
   - Learning rate optimization
   - Batch size experiments
   - Dropout rate adjustment

3. **Physics Loss**:
   - Enable physics_loss_weight > 0
   - Monitor numerical stability
   - Compare generation quality

4. **Baseline Comparison**:
   - CDVAE
   - DiffCSP
   - Other crystal generation models

## ARIS Workflow Integration

This model is ready for the ARIS research pipeline:

```bash
# Stage 1: Generate samples
python generate.py --checkpoint {results['metadata']['checkpoint']} --num_samples 100

# Stage 2: Evaluate quality
python validate.py --checkpoint {results['metadata']['checkpoint']} --split test

# Stage 3: Run ablation experiments
/run-experiment "ablation study: pure EGNN vs hybrid"

# Stage 4: Auto-review loop
/auto-review-loop "perovskite diffusion model"
```

---
*Generated by ARIS Model Analysis Pipeline*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='ARIS Model Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    results = analyze_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output
    )
    
    print("\n" + "=" * 60)
    print("Analysis complete! 🎉")
    print("=" * 60)
    print("\nRecommended next steps:")
    print("1. Generate samples: python generate.py --checkpoint <path> --num_samples 100")
    print("2. Evaluate validity: python validate.py --checkpoint <path>")
    print("3. Run ablation studies: Compare with pure EGNN/Transformer")
    print("4. Enable physics loss: Retrain with physics_loss_weight > 0")


if __name__ == '__main__':
    main()
