"""检查筛选后结构的原子数分布。"""
import json
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import Counter
from tqdm import tqdm

# 加载原始数据
with open("data/raw/perovskites.json", "r") as f:
    raw_data = json.load(f)

print(f"总共 {len(raw_data)} 个结构")

# 简单解析
structures = []
for item in raw_data[:1000]:  # 只检查前1000个
    try:
        struct_dict = item.get("structure")
        if struct_dict:
            structure = Structure.from_dict(struct_dict)
            structures.append(structure)
    except:
        pass

print(f"成功解析 {len(structures)} 个结构")

# 统计primitive cell的原子数
atom_counts = []
for struct in tqdm(structures[:100], desc="Analyzing"):  # 只分析前100个
    try:
        analyzer = SpacegroupAnalyzer(struct)
        primitive = analyzer.get_primitive_standard_structure()
        atom_counts.append(len(primitive))
    except:
        pass

# 统计分布
counter = Counter(atom_counts)
print("\nPrimitive cell原子数分布:")
for count, freq in sorted(counter.items()):
    print(f"  {count}原子: {freq}个结构")

print(f"\n5原子结构占比: {counter.get(5, 0)}/{len(atom_counts)} = {counter.get(5, 0)/len(atom_counts)*100:.1f}%")
