"""主入口CLI。"""
import argparse
import logging
import torch
import yaml
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件。"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cmd_preprocess(args):
    """预处理命令。"""
    from data.preprocess import preprocess_perovskites
    
    config = load_config(args.config) if args.config else {}
    data_config = config.get('data', {})

    preprocess_perovskites(
        raw_data_path=args.input,
        output_path=args.output,
        config=data_config
    )


def cmd_train(args):
    """训练命令。"""
    from data.dataset import get_dataloader
    from data.ionic_radii import IonicRadiiDatabase
    from models.hybrid_model import HybridEGNNTransformer
    from models.diffusion import DiffusionSchedule
    from models.physics_loss import PhysicsLoss
    from train import DiffusionTrainer
    
    config = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 数据加载
    train_loader = get_dataloader(
        config['data']['processed_data_path'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        split='train',
        augment=True
    )

    val_loader = get_dataloader(
        config['data']['processed_data_path'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        split='val',
        augment=False
    )

    # 模型 - 使用混合架构
    model = HybridEGNNTransformer(
        hidden_dim=config['model']['hidden_dim'],
        n_egnn_layers=config['model'].get('n_egnn_layers', 3),
        n_transformer_layers=config['model'].get('n_transformer_layers', 2),
        n_atom_types=config['model']['n_atom_types'],
        cutoff=config['model']['cutoff_radius'],
        num_heads=config['model'].get('num_heads', 4),
        dropout=config['model'].get('dropout', 0.1)
    )

    # 扩散调度器
    diffusion = DiffusionSchedule(
        T=config['diffusion']['timesteps'],
        schedule_type=config['diffusion']['schedule_type'],
        device=device
    )

    # 物理损失
    ionic_radii_db = IonicRadiiDatabase()
    physics_loss = PhysicsLoss(ionic_radii_db)

    # 训练器
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        physics_loss=physics_loss,
        config=config['training'],
        device=device
    )

    # 训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=config['training'].get('use_wandb', False)
    )


def cmd_generate(args):
    """生成命令。"""
    from models.egnn import EGNNModel
    from models.diffusion import DiffusionSchedule
    from generate import PerovskiteGenerator
    
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = EGNNModel(
        hidden_dim=config['model']['hidden_dim'],
        n_layers=config['model']['n_layers'],
        cutoff=config['model']['cutoff_radius'],
        n_atom_types=config['model']['n_atom_types']
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 扩散调度器
    diffusion = DiffusionSchedule(
        T=config['diffusion']['timesteps'],
        schedule_type=config['diffusion']['schedule_type'],
        device=device
    )

    # 生成器
    generator = PerovskiteGenerator(model, diffusion, config['generation'], device)

    # 原子类型（示例：BaTiO3）
    atom_types = torch.tensor([56, 22, 8, 8, 8])  # Ba, Ti, O, O, O

    # 生成
    structures = generator.generate(
        atom_types=atom_types,
        band_gap=args.band_gap,
        formation_energy=args.formation_energy,
        num_samples=args.num_samples,
        sampler=args.sampler
    )

    # 保存
    generator.save_structures(structures, args.output_dir)


def cmd_validate(args):
    """验证命令。"""
    from validate import StructureValidator
    from data.ionic_radii import IonicRadiiDatabase
    from pymatgen.core import Structure

    ionic_radii_db = IonicRadiiDatabase()
    validator = StructureValidator(ionic_radii_db, {})

    # 加载结构
    structures = []
    for cif_file in os.listdir(args.input_dir):
        if cif_file.endswith('.cif'):
            path = os.path.join(args.input_dir, cif_file)
            structure = Structure.from_file(path)
            structures.append({'structure': structure})

    # 验证
    valid_structures, report = validator.level1_geometric_filter(structures)

    logger.info(f"Validation report: {report}")
    logger.info(f"Valid structures: {len(valid_structures)}/{len(structures)}")


def main():
    parser = argparse.ArgumentParser(description="钙钛矿扩散生成系统")
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 预处理命令
    parser_preprocess = subparsers.add_parser('preprocess', help='预处理数据')
    parser_preprocess.add_argument('--input', required=True, help='输入JSON文件')
    parser_preprocess.add_argument('--output', required=True, help='输出HDF5文件')
    parser_preprocess.add_argument('--config', help='配置文件')

    # 训练命令
    parser_train = subparsers.add_parser('train', help='训练模型')
    parser_train.add_argument('--config', required=True, help='配置文件')
    parser_train.add_argument('--checkpoint-dir', default='checkpoints', help='检查点目录')
    parser_train.add_argument('--device', default='cuda', help='设备')

    # 生成命令
    parser_generate = subparsers.add_parser('generate', help='生成结构')
    parser_generate.add_argument('--config', required=True, help='配置文件')
    parser_generate.add_argument('--checkpoint', required=True, help='模型检查点')
    parser_generate.add_argument('--band-gap', type=float, default=3.0, help='目标带隙')
    parser_generate.add_argument('--formation-energy', type=float, default=-5.0, help='目标形成能')
    parser_generate.add_argument('--num-samples', type=int, default=10, help='生成样本数')
    parser_generate.add_argument('--sampler', default='ddpm', choices=['ddpm', 'ddim'], help='采样器')
    parser_generate.add_argument('--output-dir', default='outputs/generated', help='输出目录')
    parser_generate.add_argument('--device', default='cuda', help='设备')

    # 验证命令
    parser_validate = subparsers.add_parser('validate', help='验证结构')
    parser_validate.add_argument('--input-dir', required=True, help='输入CIF目录')

    args = parser.parse_args()

    if args.command == 'preprocess':
        cmd_preprocess(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'validate':
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
