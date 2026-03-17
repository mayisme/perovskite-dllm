# ML Writing Assistant - 使用指南

机器学习论文写作助手，从你的修改中学习，提供个性化的写作建议。

## 🚀 快速开始

### 安装

无需额外依赖，使用Python标准库即可运行：

```bash
cd /Users/xiaoyf/Documents/Python/dllm-perovskite/tools
python ml_writing_assistant.py --help
```

### 基础用法

**1. 获取写作建议**

```bash
python ml_writing_assistant.py \
  --action suggest \
  --text "The model achieves good performance on the test set." \
  --section Results
```

输出示例：
```
发现 2 个改进建议：

1. [quantify] 考虑量化 "good"，例如：具体的百分比、数值或统计结果
   位置: good
   优先级: high
   
2. [precision] "the test set" 过于模糊，考虑使用具体数字
   位置: test set
   优先级: medium
```

**2. 评估写作质量**

```bash
python ml_writing_assistant.py \
  --action evaluate \
  --text "Our hybrid EGNN-Transformer model achieves 87.3% validity on 100 generated samples, outperforming CDVAE (65%) and DiffCSP (70%)."
```

输出示例：
```
写作质量评估：

量化程度: 100.00%
精确性:   95.00%
正式程度: 90.00%
引用完整: 80.00%
清晰度:   85.00%

总体评分: 90.00%
```

**3. 从修改中学习**

```bash
python ml_writing_assistant.py \
  --action learn \
  --original "The model performs well" \
  --revised "The model achieves 87.3% validity" \
  --feedback "添加具体数字" \
  --section Results
```

**4. 查看统计信息**

```bash
python ml_writing_assistant.py --action stats
```

## 📚 Python API

### 基础使用

```python
from ml_writing_assistant import MLWritingAssistant

# 创建助手实例
assistant = MLWritingAssistant(workspace_dir=".")

# 获取改进建议
text = "The model shows good performance."
suggestions = assistant.suggest_improvements(text, section_type="Results")

for sug in suggestions:
    print(f"{sug['type']}: {sug['suggestion']}")
```

### 从修改中学习

```python
# 记录你的修改
assistant.learn_from_feedback(
    original="The model performs well",
    revised="The model achieves 87.3% validity",
    feedback="添加具体数字",
    section_type="Results"
)

# 助手会学习这个模式，下次遇到类似情况会提供建议
```

### 评估质量

```python
text = """
Our hybrid EGNN-Transformer model achieves 87.3% validity 
on 100 generated perovskite samples, outperforming CDVAE (65%) 
and DiffCSP (70%) by significant margins.
"""

score = assistant.evaluate_quality(text, section_type="Results")
print(f"Overall score: {score['overall']:.2%}")
```

## 🎯 核心功能

### 1. 量化检测

自动识别需要量化的模糊表述：

**触发词**：
- good, bad, high, low, many, few, large, small
- better, worse, higher, lower, more, less
- significantly, substantially, considerably

**示例**：
- ❌ "The model shows good performance"
- ✅ "The model achieves 87.3% validity"

### 2. 精确性检查

识别过于模糊的表述：

**触发词**：
- some, several, various, multiple, numerous
- approximately, roughly, about, around
- often, sometimes, usually, frequently

**示例**：
- ❌ "We conducted several experiments"
- ✅ "We conducted 18 ablation studies"

### 3. 学术化建议

检测口语化表述：

**触发词**：
- very, really, quite, pretty
- a lot of, lots of
- kind of, sort of
- get, got, gotten

**示例**：
- ❌ "We got very good results"
- ✅ "We obtained excellent results"

### 4. 引用提醒

识别需要引用的陈述：

**触发词**：
- previous work, prior research, recent studies
- it has been shown, it is known
- researchers have, studies have

**示例**：
- ❌ "Recent work has shown that..."
- ✅ "Recent work (Zhang et al., 2025) has shown that..."

### 5. 模式学习

从你的修改中学习个人风格：

```python
# 第1次修改
assistant.learn_from_feedback(
    original="good performance",
    revised="87.3% validity",
    feedback="量化"
)

# 第2次修改
assistant.learn_from_feedback(
    original="high accuracy",
    revised="92.1% accuracy",
    feedback="量化"
)

# 之后遇到 "excellent results" 时，助手会建议：
# "根据你的写作习惯，建议添加具体百分比"
```

## 📊 章节类型

助手针对不同章节提供定制化建议：

- **Abstract**: 简洁、量化、无引用
- **Introduction**: 引用完整、背景清晰
- **Method**: 精确、可复现
- **Results**: 高度量化、对比清晰
- **Discussion**: 分析深入、引用支持

## 🔧 高级用法

### 批量处理

```python
# 处理整篇论文
sections = {
    "Abstract": "...",
    "Introduction": "...",
    "Method": "...",
    "Results": "...",
    "Discussion": "..."
}

for section_type, text in sections.items():
    suggestions = assistant.suggest_improvements(text, section_type)
    print(f"\n{section_type}: {len(suggestions)} suggestions")
    for sug in suggestions[:3]:  # 显示前3个
        print(f"  - {sug['suggestion']}")
```

### 与论文大纲集成

```python
# 读取论文大纲
with open("experiments/overnight_run/paper_outline.md") as f:
    outline = f.read()

# 提取各章节
sections = extract_sections(outline)

# 逐章节改进
for section_name, content in sections.items():
    suggestions = assistant.suggest_improvements(content, section_name)
    # 应用建议...
```

### 训练数据收集

```python
# 从Git历史中提取修改
import subprocess

# 获取最近的提交
result = subprocess.run(
    ["git", "log", "--patch", "-n", "10", "paper.md"],
    capture_output=True, text=True
)

# 解析diff，提取before/after
# 自动学习你的修改模式
```

## 📈 质量评分标准

| 指标 | 权重 | 说明 |
|------|------|------|
| 量化程度 | 30% | 有数字的句子比例 |
| 精确性 | 20% | 模糊词汇的反比 |
| 正式程度 | 15% | 口语化词汇的反比 |
| 引用完整性 | 20% | 引用覆盖率 |
| 清晰度 | 15% | 句子长度适中性 |

**评分解读**：
- 90%+: 优秀，可以发表
- 80-90%: 良好，小幅改进
- 70-80%: 中等，需要改进
- <70%: 需要大幅修改

## 💡 最佳实践

### 1. 持续学习

每次修改论文后，记录你的改进：

```bash
python ml_writing_assistant.py \
  --action learn \
  --original "原文" \
  --revised "改进后" \
  --feedback "改进原因"
```

### 2. 定期评估

写完一段后立即评估：

```bash
python ml_writing_assistant.py \
  --action evaluate \
  --text "$(cat section.txt)"
```

### 3. 迭代改进

```python
# 第1轮：获取建议
suggestions = assistant.suggest_improvements(text)

# 第2轮：应用建议后重新评估
improved_text = apply_suggestions(text, suggestions)
new_score = assistant.evaluate_quality(improved_text)

# 第3轮：继续改进直到满意
```

### 4. 团队协作

```python
# 学习导师的修改
assistant.learn_from_feedback(
    original=my_draft,
    revised=advisor_revision,
    feedback="导师修改",
    section_type="Introduction"
)

# 下次写作时应用导师的风格
```

## 🐛 故障排除

### 建议太多

```python
# 只显示高优先级建议
suggestions = assistant.suggest_improvements(text)
high_priority = [s for s in suggestions if s['priority'] == 'high']
```

### 误报

```python
# 某些专业术语可能被误判，可以添加白名单
# 修改 ml_writing_assistant.py 中的规则
```

### 学习效果不明显

```python
# 需要更多训练数据（至少10-20个示例）
stats = assistant.get_statistics()
print(f"当前训练数据: {stats['total_feedback']}")
```

## 📝 示例工作流

### 撰写Results章节

```python
# 1. 写初稿
draft = """
The model performs well on the test set. 
We conducted several experiments and got good results.
"""

# 2. 获取建议
suggestions = assistant.suggest_improvements(draft, "Results")
# 输出: 需要量化 "well", "several", "good"

# 3. 改进
improved = """
The model achieves 87.3% validity on 100 test samples.
We conducted 18 ablation studies and achieved 5.2% improvement 
over the baseline.
"""

# 4. 评估
score = assistant.evaluate_quality(improved, "Results")
# 输出: Overall score: 92.00%

# 5. 学习
assistant.learn_from_feedback(draft, improved, "量化结果", "Results")
```

## 🔗 集成到工作流

### 与Git集成

```bash
# Git pre-commit hook
#!/bin/bash
python tools/ml_writing_assistant.py \
  --action evaluate \
  --text "$(cat paper.md)" > quality_report.txt

# 如果评分低于80%，警告
```

### 与CI/CD集成

```yaml
# .github/workflows/paper-quality.yml
name: Paper Quality Check
on: [push]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check writing quality
        run: |
          python tools/ml_writing_assistant.py \
            --action evaluate \
            --text "$(cat paper.md)"
```

---

**开始使用**：
```bash
cd /Users/xiaoyf/Documents/Python/dllm-perovskite/tools
python ml_writing_assistant.py --action stats
```

**需要帮助**？查看示例或提issue！
