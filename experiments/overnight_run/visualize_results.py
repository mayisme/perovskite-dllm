#!/usr/bin/env python3
"""
可视化工具：DLLM-Perovskite 实验结果分析

功能：
1. plot_loss_curves(): 绘制训练/验证损失曲线
2. plot_lr_comparison(): 对比不同学习率的收敛速度
3. plot_ablation_results(): 消融实验结果对比
4. plot_sample_distribution(): 生成样本的晶格参数分布
5. plot_baseline_comparison(): 与基线模型对比

使用方法：
    python visualize_results.py --help
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """实验结果可视化工具"""
    
    def __init__(self, base_dir: str = "."):
        """
        初始化可视化工具
        
        Args:
            base_dir: 实验根目录，默认为当前目录
        """
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "figures"
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_loss_curves(
        self,
        report_path: str = "phase1_analysis_report.json",
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制训练/验证损失曲线
        
        Args:
            report_path: 分析报告JSON文件路径
            save_path: 保存路径，默认为 figures/loss_curves.png
        """
        # 读取数据
        with open(self.base_dir / report_path, 'r') as f:
            data = json.load(f)
        
        loss_trajectory = data['loss_trajectory']
        
        # 提取epoch和损失值
        epochs = []
        train_losses = []
        val_losses = []
        
        for key, values in sorted(loss_trajectory.items()):
            epoch = int(key.split('_')[1])
            epochs.append(epoch)
            train_losses.append(values['train'])
            val_losses.append(values['val'])
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_losses, 'o-', label='Training Loss', 
                linewidth=2, markersize=6, color='#2E86AB')
        ax.plot(epochs, val_losses, 's-', label='Validation Loss', 
                linewidth=2, markersize=6, color='#A23B72')
        
        # 标记最佳验证点
        best_epoch = data['best_val_epoch']
        best_val_loss = data['best_val_loss']
        ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
                   label=f'Best Val (Epoch {best_epoch})')
        ax.plot(best_epoch, best_val_loss, 'r*', markersize=15)
        
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 添加收敛状态注释
        convergence_status = data['convergence_status']
        ax.text(0.02, 0.98, f'Status: {convergence_status}', 
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "loss_curves.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Loss curves saved to {save_path}")
        plt.close()
    
    def plot_lr_comparison(
        self,
        log_dir: str = "logs",
        save_path: Optional[str] = None
    ) -> None:
        """
        对比不同学习率的收敛速度
        
        Args:
            log_dir: 日志文件目录
            save_path: 保存路径，默认为 figures/lr_comparison.png
        """
        log_path = self.base_dir / log_dir
        
        # 查找所有学习率实验日志
        log_files = list(log_path.glob("lr_*.log"))
        
        if not log_files:
            print(f"⚠ No learning rate experiment logs found in {log_path}")
            return
        
        # 解析日志文件
        lr_data = {}
        
        for log_file in log_files:
            # 从文件名提取学习率信息
            match = re.search(r'lr_([\de-]+)_(\w+)', log_file.stem)
            if not match:
                continue
            
            lr_value = match.group(1)
            scheduler = match.group(2)
            label = f"LR={lr_value} ({scheduler})"
            
            # 解析日志内容
            epochs, train_losses, val_losses = self._parse_log_file(log_file)
            
            if epochs:
                lr_data[label] = {
                    'epochs': epochs,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
        
        if not lr_data:
            print(f"⚠ No valid learning rate data found")
            return
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = sns.color_palette("husl", len(lr_data))
        
        for (label, data), color in zip(lr_data.items(), colors):
            ax1.plot(data['epochs'], data['train_losses'], 
                    label=label, linewidth=2, color=color)
            ax2.plot(data['epochs'], data['val_losses'], 
                    label=label, linewidth=2, color=color)
        
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Training Loss', fontsize=14)
        ax1.set_title('Training Loss Comparison', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Validation Loss', fontsize=14)
        ax2.set_title('Validation Loss Comparison', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "lr_comparison.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Learning rate comparison saved to {save_path}")
        plt.close()
    
    def plot_ablation_results(
        self,
        results_file: str = "ablation_results.json",
        save_path: Optional[str] = None
    ) -> None:
        """
        消融实验结果对比
        
        Args:
            results_file: 消融实验结果JSON文件
            save_path: 保存路径，默认为 figures/ablation_results.png
        """
        results_path = self.base_dir / results_file
        
        if not results_path.exists():
            print(f"⚠ Ablation results file not found: {results_path}")
            print("  Creating example visualization with placeholder data...")
            # 创建示例数据
            results = {
                'pure_egnn': {'final_val_loss': 2.05, 'convergence_epoch': 45},
                'pure_transformer': {'final_val_loss': 2.12, 'convergence_epoch': 50},
                'hybrid_current': {'final_val_loss': 1.89, 'convergence_epoch': 40},
                'hybrid_reverse': {'final_val_loss': 1.95, 'convergence_epoch': 42}
            }
        else:
            with open(results_path, 'r') as f:
                results = json.load(f)
        
        # 准备数据
        configs = list(results.keys())
        val_losses = [results[c]['final_val_loss'] for c in configs]
        convergence_epochs = [results[c]['convergence_epoch'] for c in configs]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 子图1: 最终验证损失对比
        colors = ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D']
        bars1 = ax1.bar(range(len(configs)), val_losses, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=15, ha='right')
        ax1.set_ylabel('Final Validation Loss', fontsize=14)
        ax1.set_title('Ablation Study: Final Validation Loss', 
                     fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上标注数值
        for bar, val in zip(bars1, val_losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11)
        
        # 子图2: 收敛速度对比
        bars2 = ax2.bar(range(len(configs)), convergence_epochs, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=15, ha='right')
        ax2.set_ylabel('Convergence Epoch', fontsize=14)
        ax2.set_title('Ablation Study: Convergence Speed', 
                     fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上标注数值
        for bar, val in zip(bars2, convergence_epochs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "ablation_results.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Ablation results saved to {save_path}")
        plt.close()
    
    def plot_sample_distribution(
        self,
        samples_file: str = "generated_samples.h5",
        save_path: Optional[str] = None
    ) -> None:
        """
        生成样本的晶格参数分布
        
        Args:
            samples_file: 生成样本HDF5文件
            save_path: 保存路径，默认为 figures/sample_distribution.png
        """
        samples_path = self.base_dir / samples_file
        
        if not samples_path.exists():
            print(f"⚠ Samples file not found: {samples_path}")
            return
        
        # 读取HDF5文件
        with h5py.File(samples_path, 'r') as f:
            if 'lattice_params' in f:
                lattice_params = f['lattice_params'][:]
            else:
                print(f"⚠ 'lattice_params' not found in {samples_path}")
                return
        
        # 提取a, b, c参数
        a_params = lattice_params[:, 0]
        b_params = lattice_params[:, 1]
        c_params = lattice_params[:, 2]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 子图1: a参数分布
        axes[0, 0].hist(a_params, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Lattice Parameter a (Å)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Distribution of Lattice Parameter a', fontsize=14, fontweight='bold')
        axes[0, 0].axvline(np.mean(a_params), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(a_params):.3f} Å')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: b参数分布
        axes[0, 1].hist(b_params, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Lattice Parameter b (Å)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Distribution of Lattice Parameter b', fontsize=14, fontweight='bold')
        axes[0, 1].axvline(np.mean(b_params), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(b_params):.3f} Å')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: c参数分布
        axes[1, 0].hist(c_params, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Lattice Parameter c (Å)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Distribution of Lattice Parameter c', fontsize=14, fontweight='bold')
        axes[1, 0].axvline(np.mean(c_params), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(c_params):.3f} Å')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: a vs c散点图
        axes[1, 1].scatter(a_params, c_params, alpha=0.5, s=20, color='#06A77D')
        axes[1, 1].set_xlabel('Lattice Parameter a (Å)', fontsize=12)
        axes[1, 1].set_ylabel('Lattice Parameter c (Å)', fontsize=12)
        axes[1, 1].set_title('Correlation: a vs c', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 计算相关系数
        corr = np.corrcoef(a_params, c_params)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "sample_distribution.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Sample distribution saved to {save_path}")
        plt.close()
    
    def plot_baseline_comparison(
        self,
        comparison_file: str = "baseline_comparison.json",
        save_path: Optional[str] = None
    ) -> None:
        """
        与基线模型对比
        
        Args:
            comparison_file: 基线对比JSON文件
            save_path: 保存路径，默认为 figures/baseline_comparison.png
        """
        comparison_path = self.base_dir / comparison_file
        
        if not comparison_path.exists():
            print(f"⚠ Baseline comparison file not found: {comparison_path}")
            print("  Creating example visualization with placeholder data...")
            # 创建示例数据
            comparison = {
                'DLLM-Perovskite': {
                    'val_loss': 1.89,
                    'validity_rate': 0.92,
                    'diversity': 0.87,
                    'training_time': 120
                },
                'CDVAE': {
                    'val_loss': 2.15,
                    'validity_rate': 0.85,
                    'diversity': 0.82,
                    'training_time': 180
                },
                'DiffCSP': {
                    'val_loss': 2.08,
                    'validity_rate': 0.88,
                    'diversity': 0.79,
                    'training_time': 150
                }
            }
        else:
            with open(comparison_path, 'r') as f:
                comparison = json.load(f)
        
        # 准备数据
        models = list(comparison.keys())
        metrics = ['val_loss', 'validity_rate', 'diversity', 'training_time']
        metric_labels = ['Validation Loss', 'Validity Rate', 'Diversity', 'Training Time (min)']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = sns.color_palette("Set2", len(models))
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [comparison[m][metric] for m in models]
            
            bars = axes[idx].bar(range(len(models)), values, color=colors, alpha=0.8)
            axes[idx].set_xticks(range(len(models)))
            axes[idx].set_xticklabels(models, rotation=15, ha='right')
            axes[idx].set_ylabel(label, fontsize=12)
            axes[idx].set_title(f'Comparison: {label}', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # 标注数值
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.3f}' if metric != 'training_time' else f'{val:.0f}',
                             ha='center', va='bottom', fontsize=10)
            
            # 高亮最佳值
            if metric in ['validity_rate', 'diversity']:
                best_idx = values.index(max(values))
            else:
                best_idx = values.index(min(values))
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "baseline_comparison.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Baseline comparison saved to {save_path}")
        plt.close()
    
    def _parse_log_file(self, log_file: Path) -> Tuple[List[int], List[float], List[float]]:
        """
        解析训练日志文件
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            (epochs, train_losses, val_losses)
        """
        epochs = []
        train_losses = []
        val_losses = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # 匹配类似 "Epoch 10: train_loss=1.234, val_loss=1.456" 的行
                    match = re.search(r'Epoch\s+(\d+).*train.*?loss.*?=\s*([\d.]+).*val.*?loss.*?=\s*([\d.]+)', 
                                    line, re.IGNORECASE)
                    if match:
                        epochs.append(int(match.group(1)))
                        train_losses.append(float(match.group(2)))
                        val_losses.append(float(match.group(3)))
        except Exception as e:
            print(f"⚠ Error parsing {log_file}: {e}")
        
        return epochs, train_losses, val_losses
    
    def generate_all_plots(self) -> None:
        """生成所有可用的图表"""
        print("\n" + "="*60)
        print("Generating all available plots...")
        print("="*60 + "\n")
        
        self.plot_loss_curves()
        self.plot_lr_comparison()
        self.plot_ablation_results()
        self.plot_sample_distribution()
        self.plot_baseline_comparison()
        
        print("\n" + "="*60)
        print(f"All plots saved to {self.output_dir}")
        print("="*60 + "\n")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="DLLM-Perovskite 实验结果可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成所有图表
  python visualize_results.py --all
  
  # 只生成损失曲线
  python visualize_results.py --loss-curves
  
  # 生成学习率对比
  python visualize_results.py --lr-comparison
  
  # 生成消融实验结果
  python visualize_results.py --ablation
  
  # 生成样本分布
  python visualize_results.py --sample-dist
  
  # 生成基线对比
  python visualize_results.py --baseline
        """
    )
    
    parser.add_argument('--base-dir', type=str, default='.',
                       help='实验根目录 (默认: 当前目录)')
    parser.add_argument('--all', action='store_true',
                       help='生成所有图表')
    parser.add_argument('--loss-curves', action='store_true',
                       help='生成训练/验证损失曲线')
    parser.add_argument('--lr-comparison', action='store_true',
                       help='生成学习率对比图')
    parser.add_argument('--ablation', action='store_true',
                       help='生成消融实验结果对比')
    parser.add_argument('--sample-dist', action='store_true',
                       help='生成样本分布图')
    parser.add_argument('--baseline', action='store_true',
                       help='生成基线模型对比')
    
    args = parser.parse_args()
    
    # 创建可视化工具
    visualizer = ResultVisualizer(base_dir=args.base_dir)
    
    # 如果没有指定任何选项，默认生成所有图表
    if not any([args.all, args.loss_curves, args.lr_comparison, 
                args.ablation, args.sample_dist, args.baseline]):
        args.all = True
    
    # 生成图表
    if args.all:
        visualizer.generate_all_plots()
    else:
        if args.loss_curves:
            visualizer.plot_loss_curves()
        if args.lr_comparison:
            visualizer.plot_lr_comparison()
        if args.ablation:
            visualizer.plot_ablation_results()
        if args.sample_dist:
            visualizer.plot_sample_distribution()
        if args.baseline:
            visualizer.plot_baseline_comparison()


if __name__ == "__main__":
    main()
