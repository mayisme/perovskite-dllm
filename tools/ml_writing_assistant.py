#!/usr/bin/env python3
"""
ML Writing Assistant - 基于强化学习的机器学习论文写作助手

从用户的修改中学习写作风格，提供个性化的写作建议。
特别优化用于机器学习和材料科学论文。

Author: 小爪🐾
License: MIT
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class WritingFeedback:
    """写作反馈记录"""
    original: str
    revised: str
    feedback: str
    section_type: str  # Abstract, Introduction, Method, Results, Discussion
    timestamp: str
    improvement_type: str  # quantify, clarify, simplify, formalize
    
    def to_dict(self):
        return asdict(self)


@dataclass
class WritingPattern:
    """学习到的写作模式"""
    pattern_type: str
    examples: List[Tuple[str, str]]  # (before, after)
    frequency: int
    confidence: float
    
    def to_dict(self):
        return {
            'pattern_type': self.pattern_type,
            'examples': self.examples,
            'frequency': self.frequency,
            'confidence': self.confidence
        }


class MLWritingAssistant:
    """机器学习论文写作助手"""
    
    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir)
        self.feedback_file = self.workspace_dir / "writing_feedback.jsonl"
        self.patterns_file = self.workspace_dir / "writing_patterns.json"
        self.style_profile = self.workspace_dir / "writing_style_profile.json"
        
        # 加载历史数据
        self.feedback_history = self._load_feedback()
        self.learned_patterns = self._load_patterns()
        self.style_preferences = self._load_style_profile()
        
        # 初始化规则库
        self._init_rules()
    
    def _init_rules(self):
        """初始化写作规则库"""
        # 量化规则：识别需要量化的模糊表述
        self.quantify_triggers = [
            r'\b(good|bad|high|low|many|few|large|small|fast|slow)\b',
            r'\b(better|worse|higher|lower|more|less|larger|smaller|faster|slower)\b',
            r'\b(significantly|substantially|considerably|notably)\b',
            r'\b(recent|latest|state-of-the-art|advanced)\b'
        ]
        
        # 精确性规则：需要具体化的表述
        self.precision_triggers = [
            r'\b(some|several|various|multiple|numerous)\b',
            r'\b(approximately|roughly|about|around)\b',
            r'\b(often|sometimes|usually|frequently)\b'
        ]
        
        # 学术化规则：口语化表述
        self.formalize_triggers = [
            r'\b(very|really|quite|pretty)\b',
            r'\b(a lot of|lots of)\b',
            r'\b(kind of|sort of)\b',
            r'\b(get|got|gotten)\b'
        ]
        
        # 引用规则：需要引用的陈述
        self.citation_triggers = [
            r'\b(previous work|prior research|recent studies)\b',
            r'\b(it has been shown|it is known|it is well-established)\b',
            r'\b(researchers have|studies have|experiments have)\b'
        ]
    
    def learn_from_feedback(self, original: str, revised: str, 
                           feedback: str, section_type: str = "General"):
        """从用户修改中学习"""
        # 识别改进类型
        improvement_type = self._identify_improvement_type(original, revised)
        
        # 创建反馈记录
        fb = WritingFeedback(
            original=original,
            revised=revised,
            feedback=feedback,
            section_type=section_type,
            timestamp=datetime.now().isoformat(),
            improvement_type=improvement_type
        )
        
        # 保存反馈
        self._save_feedback(fb)
        
        # 更新模式库
        self._update_patterns(fb)
        
        # 更新风格档案
        self._update_style_profile(fb)
        
        print(f"✅ 学习完成：{improvement_type}")
        print(f"   原文: {original[:50]}...")
        print(f"   改进: {revised[:50]}...")
    
    def _identify_improvement_type(self, original: str, revised: str) -> str:
        """识别改进类型"""
        # 检查是否添加了数字
        if re.search(r'\d+\.?\d*%?', revised) and not re.search(r'\d+\.?\d*%?', original):
            return "quantify"
        
        # 检查是否添加了引用
        if re.search(r'\([A-Z][a-z]+ et al\.|[A-Z][a-z]+ \d{4}\)', revised):
            return "cite"
        
        # 检查是否简化了
        if len(revised) < len(original) * 0.8:
            return "simplify"
        
        # 检查是否更正式了
        formal_words = ['demonstrate', 'investigate', 'utilize', 'implement']
        if any(word in revised.lower() for word in formal_words):
            return "formalize"
        
        # 检查是否更清晰了
        if len(revised.split()) > len(original.split()):
            return "clarify"
        
        return "other"
    
    def suggest_improvements(self, text: str, section_type: str = "General") -> List[Dict]:
        """提供改进建议"""
        suggestions = []
        
        # 1. 检查量化机会
        for pattern in self.quantify_triggers:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._get_context(text, match.start(), match.end())
                suggestions.append({
                    'type': 'quantify',
                    'position': (match.start(), match.end()),
                    'text': match.group(),
                    'context': context,
                    'suggestion': f'考虑量化 "{match.group()}"，例如：具体的百分比、数值或统计结果',
                    'priority': 'high',
                    'examples': self._get_similar_examples('quantify')
                })
        
        # 2. 检查精确性
        for pattern in self.precision_triggers:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._get_context(text, match.start(), match.end())
                suggestions.append({
                    'type': 'precision',
                    'position': (match.start(), match.end()),
                    'text': match.group(),
                    'context': context,
                    'suggestion': f'"{match.group()}" 过于模糊，考虑使用具体数字',
                    'priority': 'medium',
                    'examples': self._get_similar_examples('clarify')
                })
        
        # 3. 检查学术化程度
        for pattern in self.formalize_triggers:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._get_context(text, match.start(), match.end())
                suggestions.append({
                    'type': 'formalize',
                    'position': (match.start(), match.end()),
                    'text': match.group(),
                    'context': context,
                    'suggestion': f'"{match.group()}" 过于口语化，考虑使用更正式的表述',
                    'priority': 'low',
                    'examples': self._get_similar_examples('formalize')
                })
        
        # 4. 检查引用需求
        for pattern in self.citation_triggers:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._get_context(text, match.start(), match.end())
                # 检查附近是否已有引用
                nearby_text = text[max(0, match.start()-50):min(len(text), match.end()+50)]
                if not re.search(r'\([A-Z][a-z]+ et al\.|[A-Z][a-z]+ \d{4}\)', nearby_text):
                    suggestions.append({
                        'type': 'citation',
                        'position': (match.start(), match.end()),
                        'text': match.group(),
                        'context': context,
                        'suggestion': '此处需要引用支持',
                        'priority': 'high'
                    })
        
        # 5. 应用学习到的模式
        learned_suggestions = self._apply_learned_patterns(text, section_type)
        suggestions.extend(learned_suggestions)
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
        
        return suggestions
    
    def _get_context(self, text: str, start: int, end: int, window: int = 40) -> str:
        """获取上下文"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _get_similar_examples(self, improvement_type: str, limit: int = 3) -> List[Tuple[str, str]]:
        """获取相似的改进示例"""
        examples = []
        for fb in self.feedback_history[-20:]:  # 最近20条
            if fb.improvement_type == improvement_type:
                examples.append((fb.original, fb.revised))
                if len(examples) >= limit:
                    break
        return examples
    
    def _apply_learned_patterns(self, text: str, section_type: str) -> List[Dict]:
        """应用学习到的模式"""
        suggestions = []
        
        for pattern in self.learned_patterns:
            if pattern.confidence < 0.5:
                continue
            
            # 检查是否匹配学习到的模式
            for before, after in pattern.examples:
                if before.lower() in text.lower():
                    suggestions.append({
                        'type': 'learned_pattern',
                        'text': before,
                        'suggestion': f'根据你的写作习惯，建议改为: {after}',
                        'priority': 'medium',
                        'confidence': pattern.confidence
                    })
        
        return suggestions
    
    def improve_text(self, text: str, section_type: str = "General") -> str:
        """自动改进文本（谨慎使用）"""
        improved = text
        suggestions = self.suggest_improvements(text, section_type)
        
        print(f"发现 {len(suggestions)} 个改进建议")
        print("注意：自动改进可能不准确，建议人工审查")
        
        return improved
    
    def evaluate_quality(self, text: str, section_type: str = "General") -> Dict:
        """评估写作质量"""
        score = {
            'quantification': 0.0,  # 量化程度
            'precision': 0.0,       # 精确性
            'formality': 0.0,       # 正式程度
            'citation': 0.0,        # 引用完整性
            'clarity': 0.0,         # 清晰度
            'overall': 0.0          # 总体评分
        }
        
        # 计算量化程度（有数字的句子比例）
        sentences = re.split(r'[.!?]+', text)
        quantified_sentences = sum(1 for s in sentences if re.search(r'\d+\.?\d*%?', s))
        score['quantification'] = quantified_sentences / max(len(sentences), 1)
        
        # 计算精确性（模糊词汇的反比）
        vague_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                         for pattern in self.precision_triggers)
        score['precision'] = max(0, 1 - vague_count / max(len(text.split()), 1) * 10)
        
        # 计算正式程度（口语化词汇的反比）
        informal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                            for pattern in self.formalize_triggers)
        score['formality'] = max(0, 1 - informal_count / max(len(text.split()), 1) * 10)
        
        # 计算引用完整性
        citation_needed = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                             for pattern in self.citation_triggers)
        citations_present = len(re.findall(r'\([A-Z][a-z]+ et al\.|[A-Z][a-z]+ \d{4}\)', text))
        score['citation'] = citations_present / max(citation_needed, 1) if citation_needed > 0 else 1.0
        
        # 计算清晰度（平均句子长度的反比）
        avg_sentence_length = len(text.split()) / max(len(sentences), 1)
        score['clarity'] = max(0, 1 - (avg_sentence_length - 20) / 30) if avg_sentence_length > 20 else 1.0
        
        # 总体评分
        score['overall'] = (
            score['quantification'] * 0.3 +
            score['precision'] * 0.2 +
            score['formality'] * 0.15 +
            score['citation'] * 0.2 +
            score['clarity'] * 0.15
        )
        
        return score
    
    def _save_feedback(self, feedback: WritingFeedback):
        """保存反馈到JSONL文件"""
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback.to_dict(), ensure_ascii=False) + '\n')
        self.feedback_history.append(feedback)
    
    def _load_feedback(self) -> List[WritingFeedback]:
        """加载历史反馈"""
        if not self.feedback_file.exists():
            return []
        
        feedback_list = []
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                feedback_list.append(WritingFeedback(**data))
        return feedback_list
    
    def _update_patterns(self, feedback: WritingFeedback):
        """更新模式库"""
        # 简化的模式学习：记录常见的before->after转换
        pattern_key = feedback.improvement_type
        
        # 查找或创建模式
        pattern = None
        for p in self.learned_patterns:
            if p.pattern_type == pattern_key:
                pattern = p
                break
        
        if pattern is None:
            pattern = WritingPattern(
                pattern_type=pattern_key,
                examples=[],
                frequency=0,
                confidence=0.0
            )
            self.learned_patterns.append(pattern)
        
        # 更新模式
        pattern.examples.append((feedback.original, feedback.revised))
        pattern.frequency += 1
        pattern.confidence = min(1.0, pattern.frequency / 10)  # 10次后达到满信心
        
        # 保存
        self._save_patterns()
    
    def _save_patterns(self):
        """保存模式库"""
        data = [p.to_dict() for p in self.learned_patterns]
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_patterns(self) -> List[WritingPattern]:
        """加载模式库"""
        if not self.patterns_file.exists():
            return []
        
        with open(self.patterns_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [WritingPattern(**p) for p in data]
    
    def _update_style_profile(self, feedback: WritingFeedback):
        """更新风格档案"""
        # 统计用户的写作偏好
        if feedback.improvement_type not in self.style_preferences:
            self.style_preferences[feedback.improvement_type] = 0
        self.style_preferences[feedback.improvement_type] += 1
        
        # 保存
        with open(self.style_profile, 'w', encoding='utf-8') as f:
            json.dump(self.style_preferences, f, indent=2)
    
    def _load_style_profile(self) -> Dict:
        """加载风格档案"""
        if not self.style_profile.exists():
            return {}
        
        with open(self.style_profile, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_feedback': len(self.feedback_history),
            'learned_patterns': len(self.learned_patterns),
            'improvement_types': dict(self.style_preferences),
            'recent_feedback': [fb.to_dict() for fb in self.feedback_history[-5:]]
        }


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Writing Assistant')
    parser.add_argument('--workspace', default='.', help='工作区目录')
    parser.add_argument('--action', choices=['suggest', 'evaluate', 'learn', 'stats'], 
                       required=True, help='操作类型')
    parser.add_argument('--text', help='要分析的文本')
    parser.add_argument('--section', default='General', help='章节类型')
    parser.add_argument('--original', help='原始文本（用于学习）')
    parser.add_argument('--revised', help='修改后文本（用于学习）')
    parser.add_argument('--feedback', help='反馈说明（用于学习）')
    
    args = parser.parse_args()
    
    assistant = MLWritingAssistant(args.workspace)
    
    if args.action == 'suggest':
        if not args.text:
            print("错误：需要提供 --text 参数")
            return
        
        suggestions = assistant.suggest_improvements(args.text, args.section)
        print(f"\n发现 {len(suggestions)} 个改进建议：\n")
        for i, sug in enumerate(suggestions, 1):
            print(f"{i}. [{sug['type']}] {sug['suggestion']}")
            print(f"   位置: {sug.get('text', 'N/A')}")
            print(f"   优先级: {sug.get('priority', 'N/A')}")
            if 'examples' in sug and sug['examples']:
                print(f"   示例:")
                for before, after in sug['examples'][:2]:
                    print(f"     • {before[:40]}... → {after[:40]}...")
            print()
    
    elif args.action == 'evaluate':
        if not args.text:
            print("错误：需要提供 --text 参数")
            return
        
        score = assistant.evaluate_quality(args.text, args.section)
        print("\n写作质量评估：\n")
        print(f"量化程度: {score['quantification']:.2%}")
        print(f"精确性:   {score['precision']:.2%}")
        print(f"正式程度: {score['formality']:.2%}")
        print(f"引用完整: {score['citation']:.2%}")
        print(f"清晰度:   {score['clarity']:.2%}")
        print(f"\n总体评分: {score['overall']:.2%}")
    
    elif args.action == 'learn':
        if not all([args.original, args.revised, args.feedback]):
            print("错误：学习模式需要 --original, --revised, --feedback 参数")
            return
        
        assistant.learn_from_feedback(
            args.original, args.revised, args.feedback, args.section
        )
    
    elif args.action == 'stats':
        stats = assistant.get_statistics()
        print("\n统计信息：\n")
        print(f"总反馈数: {stats['total_feedback']}")
        print(f"学习模式数: {stats['learned_patterns']}")
        print(f"\n改进类型分布:")
        for imp_type, count in stats['improvement_types'].items():
            print(f"  {imp_type}: {count}")


if __name__ == '__main__':
    main()
