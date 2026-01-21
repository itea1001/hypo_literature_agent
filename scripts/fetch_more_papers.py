#!/usr/bin/env python3
"""
Fetch more conference papers for Ryan to categorize
"""

import os
import sys
import json
from pathlib import Path

# Add the paper scraper to path
sys.path.insert(0, str(Path(__file__).parent.parent / "neurips_scraper" / "paper-scraper" / "src"))

from scrapers.openreview_scraper import ConferenceScraper

def fetch_conference_papers(conference: str, year: int, limit: int, output_dir: Path):
    """Fetch papers from a conference using the scraper"""
    scraper = ConferenceScraper()
    
    print(f"\nFetching {limit} papers from {conference.upper()} {year}...")
    papers = scraper.get_conference_papers(conference, year, limit=limit)
    
    if not papers:
        print(f"  No papers found")
        return []
    
    print(f"  Found {len(papers)} papers")
    
    # Save metadata for each paper
    results = []
    for paper in papers:
        info = scraper.extract_paper_info(paper, conference=conference.upper())
        results.append(info)
    
    # Save combined file
    combined_path = output_dir / f"{conference}_{year}_full.json"
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} papers to {combined_path}")
    
    return results

def main():
    base_dir = Path(__file__).parent.parent / "literature_review"
    conference_dir = base_dir / "papers" / "conferences"
    conference_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Fetching more conference papers for broader coverage")
    print("="*60)
    
    # Fetch more papers from major ML conferences
    # 200 from each should give good coverage
    conferences = [
        ('neurips', 2024, 200),
        ('icml', 2024, 200),
        ('iclr', 2024, 200),
    ]
    
    total = 0
    for conf, year, limit in conferences:
        results = fetch_conference_papers(conf, year, limit, conference_dir)
        total += len(results)
    
    print("\n" + "="*60)
    print(f"Done! Fetched {total} papers total")
    print(f"Saved to: {conference_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

