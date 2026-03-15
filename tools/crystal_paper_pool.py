#!/usr/bin/env python3
"""
CrystalGenPaperPool - arXiv Paper Collection Tool for Crystal Generation Research

Inspired by HandyLLMTools/GalleryToMD design philosophy.
Fetches, parses, and organizes crystal generation papers from arXiv.

Author: Generated for dllm-perovskite project
Date: 2026-03-16
"""

import json
import time
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict


@dataclass
class PaperEntry:
    """Data model for a single paper entry."""
    title: str
    authors: List[str]
    date: str
    arxiv_id: str
    abstract: str
    methods: List[str]
    datasets: List[str]
    metrics: Dict[str, str]
    architecture_type: str
    key_contributions: List[str]
    relevance_score: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class CrystalPaperPool:
    """
    Main class for fetching and organizing crystal generation papers.
    
    Features:
    - arXiv API integration with polite rate limiting
    - Keyword-based filtering and relevance scoring
    - Caching to avoid redundant requests
    - Time range filtering (2022-2026)
    - Structured output in Markdown and JSON
    """
    
    # arXiv API configuration
    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    REQUEST_DELAY = 3.0  # seconds between requests (arXiv guideline: 3s)
    
    # Search keywords for crystal generation domain
    KEYWORDS = {
        "primary": ["crystal generation", "crystal structure prediction", "materials generation"],
        "methods": ["diffusion", "CDVAE", "DiffCSP", "flow matching", "VAE", "GAN"],
        "materials": ["perovskite", "MOF", "zeolite", "inorganic crystal"],
        "datasets": ["MP-20", "JARVIS", "Materials Project", "OQMD"],
        "architectures": ["graph neural network", "GNN", "transformer", "equivariant"]
    }
    
    def __init__(self, cache_dir: str = "./cache", output_dir: str = "./output"):
        """
        Initialize the paper pool.
        
        Args:
            cache_dir: Directory for caching API responses
            output_dir: Directory for output files
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.papers: List[PaperEntry] = []
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce polite rate limiting between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _build_query(self, keywords: List[str], start_date: str, end_date: str) -> str:
        """
        Build arXiv API query string.
        
        Args:
            keywords: List of search keywords
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            
        Returns:
            Query string for arXiv API
        """
        # Combine keywords with OR logic
        keyword_query = " OR ".join([f'all:"{kw}"' for kw in keywords])
        
        # Add category filter (physics, cs, cond-mat)
        category_filter = "(cat:cond-mat.* OR cat:cs.LG OR cat:physics.comp-ph)"
        
        # Combine with date range
        query = f"({keyword_query}) AND {category_filter}"
        
        return query
    
    def fetch_papers(
        self,
        max_results: int = 100,
        start_date: Optional[str] = "2022-01-01",
        end_date: Optional[str] = None
    ) -> List[PaperEntry]:
        """
        Fetch papers from arXiv API.
        
        Args:
            max_results: Maximum number of papers to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            List of PaperEntry objects
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Build query
        all_keywords = (
            self.KEYWORDS["primary"] + 
            self.KEYWORDS["methods"] + 
            self.KEYWORDS["materials"]
        )
        query = self._build_query(all_keywords, start_date, end_date)
        
        # Check cache
        cache_key = f"{query}_{max_results}_{start_date}_{end_date}"
        cache_file = self.cache_dir / f"{hash(cache_key)}.json"
        
        if cache_file.exists():
            print(f"📦 Loading from cache: {cache_file.name}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                self.papers = [PaperEntry(**p) for p in cached_data]
                return self.papers
        
        # Fetch from arXiv
        print(f"🔍 Fetching papers from arXiv...")
        print(f"   Query: {query[:100]}...")
        print(f"   Date range: {start_date} to {end_date}")
        
        self._rate_limit()
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        url = f"{self.ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
        except Exception as e:
            print(f"❌ Error fetching from arXiv: {e}")
            return []
        
        # Parse XML response
        self.papers = self._parse_arxiv_response(xml_data, start_date, end_date)
        
        # Cache results
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in self.papers], f, indent=2, ensure_ascii=False)
        
        print(f"✅ Fetched {len(self.papers)} papers")
        return self.papers
    
    def _parse_arxiv_response(self, xml_data: str, start_date: str, end_date: str) -> List[PaperEntry]:
        """Parse arXiv API XML response into PaperEntry objects."""
        papers = []
        
        # Parse XML
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            try:
                # Extract basic metadata
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
                abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                published = entry.find('atom:published', ns).text[:10]  # YYYY-MM-DD
                
                # Filter by date range
                if published < start_date or published > end_date:
                    continue
                
                # Extract authors
                authors = [
                    author.find('atom:name', ns).text 
                    for author in entry.findall('atom:author', ns)
                ]
                
                # Extract methods, datasets, architectures from abstract
                methods = self._extract_keywords(abstract, self.KEYWORDS["methods"])
                datasets = self._extract_keywords(abstract, self.KEYWORDS["datasets"])
                architecture_type = self._extract_architecture(abstract)
                
                # Extract metrics (simple heuristic)
                metrics = self._extract_metrics(abstract)
                
                # Extract key contributions (first 3 sentences)
                key_contributions = self._extract_contributions(abstract)
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance(title, abstract)
                
                paper = PaperEntry(
                    title=title,
                    authors=authors,
                    date=published,
                    arxiv_id=arxiv_id,
                    abstract=abstract,
                    methods=methods,
                    datasets=datasets,
                    metrics=metrics,
                    architecture_type=architecture_type,
                    key_contributions=key_contributions,
                    relevance_score=relevance_score
                )
                
                papers.append(paper)
                
            except Exception as e:
                print(f"⚠️  Error parsing entry: {e}")
                continue
        
        # Sort by relevance score
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        return papers
    
    def _extract_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Extract matching keywords from text (case-insensitive)."""
        text_lower = text.lower()
        found = []
        for kw in keywords:
            if kw.lower() in text_lower:
                found.append(kw)
        return found
    
    def _extract_architecture(self, text: str) -> str:
        """Identify the main architecture type mentioned."""
        text_lower = text.lower()
        for arch in self.KEYWORDS["architectures"]:
            if arch.lower() in text_lower:
                return arch
        return "unknown"
    
    def _extract_metrics(self, text: str) -> Dict[str, str]:
        """Extract performance metrics from abstract (simple regex)."""
        metrics = {}
        
        # Common patterns: "MAE of 0.05", "accuracy: 95%", "RMSE = 0.1"
        patterns = [
            r'(MAE|RMSE|accuracy|precision|recall|F1)[:\s=]+([0-9.]+%?)',
            r'([0-9.]+%?)\s+(MAE|RMSE|accuracy)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    key, value = match if match[0].isalpha() else (match[1], match[0])
                    metrics[key.upper()] = value
        
        return metrics
    
    def _extract_contributions(self, abstract: str) -> List[str]:
        """Extract key contributions (first 3 sentences)."""
        sentences = re.split(r'[.!?]+', abstract)
        contributions = [s.strip() for s in sentences[:3] if s.strip()]
        return contributions
    
    def _calculate_relevance(self, title: str, abstract: str) -> float:
        """
        Calculate relevance score based on keyword matching.
        
        Score components:
        - Primary keywords in title: +2.0 each
        - Primary keywords in abstract: +1.0 each
        - Method keywords: +0.5 each
        - Dataset keywords: +0.3 each
        """
        score = 0.0
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        
        # Primary keywords
        for kw in self.KEYWORDS["primary"]:
            if kw.lower() in title_lower:
                score += 2.0
            elif kw.lower() in abstract_lower:
                score += 1.0
        
        # Method keywords
        for kw in self.KEYWORDS["methods"]:
            if kw.lower() in abstract_lower:
                score += 0.5
        
        # Dataset keywords
        for kw in self.KEYWORDS["datasets"]:
            if kw.lower() in abstract_lower:
                score += 0.3
        
        return round(score, 2)
    
    def export_markdown(self, filename: str = "crystal_papers.md", group_by: str = "date"):
        """
        Export papers to Markdown format.
        
        Args:
            filename: Output filename
            group_by: Grouping strategy ("date", "method", "relevance")
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Crystal Generation Papers\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"*Total papers: {len(self.papers)}*\n\n")
            f.write("---\n\n")
            
            if group_by == "date":
                self._write_by_date(f)
            elif group_by == "method":
                self._write_by_method(f)
            elif group_by == "relevance":
                self._write_by_relevance(f)
        
        print(f"📄 Markdown exported to: {output_path}")
    
    def _write_by_date(self, f):
        """Write papers grouped by year."""
        by_year = defaultdict(list)
        for paper in self.papers:
            year = paper.date[:4]
            by_year[year].append(paper)
        
        for year in sorted(by_year.keys(), reverse=True):
            f.write(f"## {year}\n\n")
            for paper in by_year[year]:
                self._write_paper_entry(f, paper)
    
    def _write_by_method(self, f):
        """Write papers grouped by method."""
        by_method = defaultdict(list)
        for paper in self.papers:
            if paper.methods:
                for method in paper.methods:
                    by_method[method].append(paper)
            else:
                by_method["Other"].append(paper)
        
        for method in sorted(by_method.keys()):
            f.write(f"## {method}\n\n")
            for paper in by_method[method]:
                self._write_paper_entry(f, paper)
    
    def _write_by_relevance(self, f):
        """Write papers sorted by relevance score."""
        f.write("## High Relevance (Score ≥ 3.0)\n\n")
        for paper in self.papers:
            if paper.relevance_score >= 3.0:
                self._write_paper_entry(f, paper)
        
        f.write("## Medium Relevance (1.0 ≤ Score < 3.0)\n\n")
        for paper in self.papers:
            if 1.0 <= paper.relevance_score < 3.0:
                self._write_paper_entry(f, paper)
        
        f.write("## Lower Relevance (Score < 1.0)\n\n")
        for paper in self.papers:
            if paper.relevance_score < 1.0:
                self._write_paper_entry(f, paper)
    
    def _write_paper_entry(self, f, paper: PaperEntry):
        """Write a single paper entry in Markdown."""
        f.write(f"### {paper.title}\n\n")
        f.write(f"**Authors:** {', '.join(paper.authors[:3])}")
        if len(paper.authors) > 3:
            f.write(f" *et al.*")
        f.write(f"\n\n")
        f.write(f"**Date:** {paper.date} | **arXiv:** [{paper.arxiv_id}](https://arxiv.org/abs/{paper.arxiv_id}) | **Relevance:** {paper.relevance_score}\n\n")
        
        if paper.methods:
            f.write(f"**Methods:** {', '.join(paper.methods)}\n\n")
        
        if paper.datasets:
            f.write(f"**Datasets:** {', '.join(paper.datasets)}\n\n")
        
        if paper.metrics:
            metrics_str = ', '.join([f"{k}: {v}" for k, v in paper.metrics.items()])
            f.write(f"**Metrics:** {metrics_str}\n\n")
        
        f.write(f"**Abstract:** {paper.abstract[:300]}...\n\n")
        
        if paper.key_contributions:
            f.write("**Key Points:**\n")
            for contrib in paper.key_contributions[:2]:
                f.write(f"- {contrib}\n")
            f.write("\n")
        
        f.write("---\n\n")
    
    def export_json(self, filename: str = "crystal_papers.json"):
        """Export papers to JSON format."""
        output_path = self.output_dir / filename
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_papers": len(self.papers),
            "papers": [p.to_dict() for p in self.papers]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 JSON exported to: {output_path}")
    
    def export_related_work_template(self, filename: str = "related_work_template.md"):
        """Generate a related work section template."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Related Work Template\n\n")
            f.write("*Auto-generated from CrystalGenPaperPool*\n\n")
            f.write("## Crystal Structure Generation\n\n")
            
            # Group by method for related work
            by_method = defaultdict(list)
            for paper in self.papers[:20]:  # Top 20 most relevant
                if paper.methods:
                    for method in paper.methods:
                        by_method[method].append(paper)
            
            for method, papers in sorted(by_method.items()):
                f.write(f"### {method}-based Approaches\n\n")
                for paper in papers[:3]:  # Top 3 per method
                    citation = f"[{paper.authors[0].split()[-1]} et al., {paper.date[:4]}]"
                    f.write(f"- {citation} proposed {paper.title}. ")
                    if paper.key_contributions:
                        f.write(f"{paper.key_contributions[0]} ")
                    f.write(f"(arXiv:{paper.arxiv_id})\n")
                f.write("\n")
        
        print(f"📝 Related work template exported to: {output_path}")
    
    def compare_with_previous(self, previous_json: str) -> Dict:
        """
        Compare current papers with a previous fetch to find new papers.
        
        Args:
            previous_json: Path to previous JSON export
            
        Returns:
            Dictionary with new, updated, and unchanged papers
        """
        previous_path = Path(previous_json)
        if not previous_path.exists():
            print(f"⚠️  Previous file not found: {previous_json}")
            return {"new": self.papers, "updated": [], "unchanged": []}
        
        with open(previous_path, 'r', encoding='utf-8') as f:
            previous_data = json.load(f)
        
        previous_ids = {p["arxiv_id"] for p in previous_data["papers"]}
        current_ids = {p.arxiv_id for p in self.papers}
        
        new_ids = current_ids - previous_ids
        unchanged_ids = current_ids & previous_ids
        
        new_papers = [p for p in self.papers if p.arxiv_id in new_ids]
        unchanged_papers = [p for p in self.papers if p.arxiv_id in unchanged_ids]
        
        print(f"📈 Comparison results:")
        print(f"   New papers: {len(new_papers)}")
        print(f"   Unchanged: {len(unchanged_papers)}")
        
        return {
            "new": new_papers,
            "updated": [],  # Could implement update detection
            "unchanged": unchanged_papers
        }


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CrystalGenPaperPool - arXiv Paper Collector")
    parser.add_argument("--max-results", type=int, default=50, help="Maximum papers to fetch")
    parser.add_argument("--start-date", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--group-by", type=str, default="relevance", 
                       choices=["date", "method", "relevance"], help="Grouping strategy")
    parser.add_argument("--compare", type=str, default=None, help="Compare with previous JSON")
    
    args = parser.parse_args()
    
    # Initialize pool
    pool = CrystalPaperPool(cache_dir=args.cache_dir, output_dir=args.output_dir)
    
    # Fetch papers
    papers = pool.fetch_papers(
        max_results=args.max_results,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if not papers:
        print("❌ No papers found")
        return
    
    # Export results
    pool.export_markdown(group_by=args.group_by)
    pool.export_json()
    pool.export_related_work_template()
    
    # Compare if requested
    if args.compare:
        pool.compare_with_previous(args.compare)
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
