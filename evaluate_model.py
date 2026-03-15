#!/usr/bin/env python3
"""
ARIS-style Model Evaluation Script
评估训练好的混合架构模型的生成质量
"""

import torch
import h5py
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import argparse

from models.hybrid_model import HybridEGNNTransformer
from data.dataset import PerovskiteDataset
from utils.config_utils import load_config


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, checkpoint_path, config_path, device='cpu'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # 加载配置
        self.config = load_config(config_path)
        
        # 加载模型
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model()
        self.model.eval()
        
        # 加载数据集
        print("Loading dataset...")
        self.dataset = PerovskiteDataset(
            self.config['data']['h5_path'],
            split='test'
        )
        
        print(f"✓ Model loaded: {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"✓ Test set: {len(self.dataset)} samples")
    
    def _load_model(self):
        """加载训练好的模型"""
        # 创建模型
        model = HybridEGNNTransformer(
            hidden_dim=self.config['model']['hidden_dim'],
            n_egnn_layers=self.config['model']['n_egnn_layers'],
            n_transformer_layers=self.config['model']['n_transformer_layers'],
            num_heads=self.config['model']['num_heads'],
            dropout=self.config['model']['dropout'],
            cutoff=self.config['model'].get('cutoff_radius', 6.0)
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def evaluate_reconstruction(self, num_samples=100):
        """评估重构质量（去噪能力）"""
        print(f"\n=== Reconstruction Evaluation ({num_samples} samples) ===")
        
        reconstruction_errors = []
        
        for i in tqdm(range(min(num_samples, len(self.dataset)))):
            data = self.dataset[i]
            
            # 转换为batch
            batch = {
                'atom_types': data['atom_types'].unsqueeze(0).to(self.device),
                'coords': data['coords'].unsqueeze(0).to(self.device),
                'lattice_params': data['lattice_params'].unsqueeze(0).to(self.device),
                'num_atoms': torch.tensor([data['num_atoms']], device=self.device)
            }
            
            with torch.no_grad():
                # 添加噪声
                t = torch.tensor([0.5], device=self.device)  # 中等噪声水平
                noise = torch.randn_like(batch['coords'])
                noisy_coords = batch['coords'] + 0.5 * noise
                
                # 预测噪声
                pred_noise = self.model(
                    batch['atom_types'],
                    noisy_coords,
                    batch['lattice_params'],
                    t,
                    batch['num_atoms']
                )
                
                # 计算重构误差
                error = torch.mean((pred_noise - noise) ** 2).item()
                reconstruction_errors.append(error)
        
        results = {
            'mean_error': np.mean(reconstruction_errors),
            'std_error': np.std(reconstruction_errors),
            'median_error': np.median(reconstruction_errors),
            'min_error': np.min(reconstruction_errors),
            'max_error': np.max(reconstruction_errors)
        }
        
        print(f"Mean reconstruction error: {results['mean_error']:.4f} ± {results['std_error']:.4f}")
        print(f"Median: {results['median_error']:.4f}, Range: [{results['min_error']:.4f}, {results['max_error']:.4f}]")
        
        return results
    
    def evaluate_lattice_prediction(self, num_samples=100):
        """评估晶格参数预测准确性"""
        print(f"\n=== Lattice Parameter Prediction ({num_samples} samples) ===")
        
        lattice_errors = {
            'a': [], 'b': [], 'c': [],
            'alpha': [], 'beta': [], 'gamma': []
        }
        
        for i in tqdm(range(min(num_samples, len(self.dataset)))):
            data = self.dataset[i]
            
            # 真实晶格参数
            true_lattice = data['lattice_params'].cpu().numpy()
            
            # 这里我们评估模型对晶格参数的敏感性
            # 通过扰动晶格参数并观察模型输出的变化
            batch = {
                'atom_types': data['atom_types'].unsqueeze(0).to(self.device),
                'coords': data['coords'].unsqueeze(0).to(self.device),
                'lattice_params': data['lattice_params'].unsqueeze(0).to(self.device),
                'num_atoms': torch.tensor([data['num_atoms']], device=self.device)
            }
            
            with torch.no_grad():
                t = torch.tensor([0.5], device=self.device)
                
                # 原始预测
                pred_original = self.model(
                    batch['atom_types'],
                    batch['coords'],
                    batch['lattice_params'],
                    t,
                    batch['num_atoms']
                )
                
                # 扰动晶格参数（±5%）
                perturbed_lattice = batch['lattice_params'].clone()
                perturbed_lattice[:, :3] *= 1.05  # a, b, c +5%
                
                pred_perturbed = self.model(
                    batch['atom_types'],
                    batch['coords'],
                    perturbed_lattice,
                    t,
                    batch['num_atoms']
                )
                
                # 计算预测差异（模型对晶格参数的敏感性）
                sensitivity = torch.mean((pred_perturbed - pred_original) ** 2).item()
                
                # 记录（这里简化为敏感性指标）
                lattice_errors['a'].append(sensitivity)
        
        results = {
            'lattice_sensitivity': {
                'mean': np.mean(lattice_errors['a']),
                'std': np.std(lattice_errors['a'])
            }
        }
        
        print(f"Lattice sensitivity: {results['lattice_sensitivity']['mean']:.6f} ± {results['lattice_sensitivity']['std']:.6f}")
        print("(Lower = model is more robust to lattice perturbations)")
        
        return results
    
    def evaluate_model_capacity(self):
        """评估模型容量和复杂度"""
        print("\n=== Model Capacity Analysis ===")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 分析各组件参数量
        egnn_params = sum(p.numel() for name, p in self.model.named_parameters() if 'egnn' in name)
        transformer_params = sum(p.numel() for name, p in self.model.named_parameters() if 'transformer' in name)
        
        results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'egnn_parameters': egnn_params,
            'transformer_parameters': transformer_params,
            'egnn_ratio': egnn_params / total_params,
            'transformer_ratio': transformer_params / total_params
        }
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"EGNN parameters: {egnn_params:,} ({results['egnn_ratio']:.1%})")
        print(f"Transformer parameters: {transformer_params:,} ({results['transformer_ratio']:.1%})")
        
        return results
    
    def run_full_evaluation(self, output_dir='evaluation_results'):
        """运行完整评估"""
        print("=" * 60)
        print("ARIS Model Evaluation Pipeline")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        results = {
            'metadata': {
                'checkpoint': str(self.checkpoint_path),
                'config': str(self.config),
                'device': self.device,
                'timestamp': datetime.now().isoformat(),
                'test_samples': len(self.dataset)
            }
        }
        
        # 1. 模型容量分析
        results['model_capacity'] = self.evaluate_model_capacity()
        
        # 2. 重构质量评估
        results['reconstruction'] = self.evaluate_reconstruction(num_samples=100)
        
        # 3. 晶格参数预测
        results['lattice_prediction'] = self.evaluate_lattice_prediction(num_samples=100)
        
        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_path / f'evaluation_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")
        
        # 生成评估报告
        self._generate_report(results, output_path / f'report_{timestamp}.md')
        
        return results
    
    def _generate_report(self, results, output_path):
        """生成Markdown评估报告"""
        report = f"""# Model Evaluation Report

**Date**: {results['metadata']['timestamp']}
**Checkpoint**: {results['metadata']['checkpoint']}
**Test Samples**: {results['metadata']['test_samples']}

## 1. Model Capacity

- **Total Parameters**: {results['model_capacity']['total_parameters']:,}
- **EGNN Parameters**: {results['model_capacity']['egnn_parameters']:,} ({results['model_capacity']['egnn_ratio']:.1%})
- **Transformer Parameters**: {results['model_capacity']['transformer_parameters']:,} ({results['model_capacity']['transformer_ratio']:.1%})

## 2. Reconstruction Quality

- **Mean Error**: {results['reconstruction']['mean_error']:.4f} ± {results['reconstruction']['std_error']:.4f}
- **Median Error**: {results['reconstruction']['median_error']:.4f}
- **Range**: [{results['reconstruction']['min_error']:.4f}, {results['reconstruction']['max_error']:.4f}]

## 3. Lattice Sensitivity

- **Sensitivity**: {results['lattice_prediction']['lattice_sensitivity']['mean']:.6f} ± {results['lattice_prediction']['lattice_sensitivity']['std']:.6f}

## Summary

The hybrid EGNN-Transformer model shows:
- Balanced architecture with {results['model_capacity']['egnn_ratio']:.1%} EGNN and {results['model_capacity']['transformer_ratio']:.1%} Transformer
- Reconstruction error: {results['reconstruction']['mean_error']:.4f} (lower is better)
- Lattice sensitivity: {results['lattice_prediction']['lattice_sensitivity']['mean']:.6f} (lower = more robust)

## Next Steps

1. **Generate samples** using `generate.py`
2. **Evaluate validity** (crystal structure constraints)
3. **Assess diversity** (coverage of chemical space)
4. **Compare with baselines** (CDVAE, DiffCSP)
5. **Enable physics loss** and retrain
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ARIS Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # 运行评估
    results = evaluator.run_full_evaluation(output_dir=args.output)
    
    print("\n" + "=" * 60)
    print("Evaluation complete! 🎉")
    print("=" * 60)


if __name__ == '__main__':
    main()
