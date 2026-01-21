#!/usr/bin/env python3
"""
Script to fetch literature for hypothesis generation project
Uses the paper scraper for conference papers and adds arxiv support
"""

import os
import sys
import json
import requests
import time
from pathlib import Path

# Add the paper scraper to path
sys.path.insert(0, str(Path(__file__).parent.parent / "neurips_scraper" / "paper-scraper" / "src"))

from scrapers.openreview_scraper import ConferenceScraper

# Arxiv papers for hypothesis generation literature review
ARXIV_PAPERS = {
    "2410.07076": "MOOSE-Chem",
    "2504.14191": "AI-Idea-Bench-2025", 
    "2504.11524": "HypoBench",
    "2302.12832": "Fluid-Transformers-Creative-Analogies",
    "2206.01328": "Augmenting-Scientific-Creativity-Retrieval",
    "2202.12826": "Idea-Mining-Survey",
    "2402.08565": "AI-Literature-Reviews",
    "2402.14978": "AI-Augmented-Brainwriting",
}

def fetch_arxiv_paper(arxiv_id: str, output_dir: Path, name: str = None):
    """Download arxiv paper PDF and metadata"""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    
    filename = name or arxiv_id
    pdf_path = output_dir / f"{filename}.pdf"
    meta_path = output_dir / f"{filename}.json"
    
    # Download PDF
    print(f"Downloading {arxiv_id} -> {pdf_path}")
    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"  PDF saved ({len(response.content)} bytes)")
    except Exception as e:
        print(f"  Error downloading PDF: {e}")
        return False
    
    # Get metadata from arxiv API
    time.sleep(0.5)  # Rate limit
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        # Parse basic info from XML (simplified)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entry = root.find('atom:entry', ns)
        if entry is not None:
            title = entry.find('atom:title', ns)
            abstract = entry.find('atom:summary', ns)
            authors = entry.findall('atom:author/atom:name', ns)
            
            metadata = {
                'arxiv_id': arxiv_id,
                'title': title.text.strip() if title is not None else '',
                'abstract': abstract.text.strip() if abstract is not None else '',
                'authors': [a.text for a in authors],
                'pdf_url': pdf_url,
                'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}"
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  Metadata saved: {metadata['title'][:60]}...")
            
    except Exception as e:
        print(f"  Error fetching metadata: {e}")
    
    return True

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
        
        # Save individual paper metadata
        safe_title = "".join(c for c in info['title'][:50] if c.isalnum() or c in ' -_').strip()
        meta_path = output_dir / f"{conference}_{year}_{safe_title}.json"
        with open(meta_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    # Save combined file
    combined_path = output_dir / f"{conference}_{year}_papers.json"
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved combined metadata to {combined_path}")
    
    return results

def main():
    base_dir = Path(__file__).parent.parent / "literature_review"
    
    # Create output directories
    arxiv_dir = base_dir / "papers" / "arxiv"
    conference_dir = base_dir / "papers" / "conferences"
    arxiv_dir.mkdir(parents=True, exist_ok=True)
    conference_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Fetching hypothesis generation literature from arxiv")
    print("="*60)
    
    for arxiv_id, name in ARXIV_PAPERS.items():
        fetch_arxiv_paper(arxiv_id, arxiv_dir, name)
        time.sleep(1)  # Be nice to arxiv
    
    print("\n" + "="*60)
    print("Fetching sample conference papers using paper scraper")
    print("="*60)
    
    # Fetch sample papers from major ML conferences
    conferences = [
        ('neurips', 2024, 20),
        ('icml', 2024, 20),
        ('iclr', 2024, 20),
    ]
    
    for conf, year, limit in conferences:
        fetch_conference_papers(conf, year, limit, conference_dir)
        time.sleep(2)
    
    print("\n" + "="*60)
    print("Done! Papers saved to:")
    print(f"  Arxiv: {arxiv_dir}")
    print(f"  Conferences: {conference_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

