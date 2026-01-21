"""
Single-domain research idea generator.

This module generates research ideas within a single domain by:
1. Loading papers from a specific category
2. Extracting key concepts, methods, and gaps
3. Generating novel research ideas based on the literature
4. Optionally self-critiquing and refining ideas
"""

import json
import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a paper from our corpus."""
    id: str
    title: str
    conference: str
    keywords: List[str]
    abstract: Optional[str] = None


@dataclass 
class ResearchIdea:
    """Represents a generated research idea."""
    id: str
    title: str
    description: str
    motivation: str
    source_papers: List[str]  # Paper IDs that inspired this idea
    domain: str
    generated_at: str
    confidence_score: Optional[float] = None
    critique: Optional[str] = None
    refined_version: Optional[str] = None


class SingleDomainGenerator:
    """
    Generates research ideas within a single domain.
    
    The generation process:
    1. Load papers from the specified category
    2. Extract common themes, methods, and potential gaps
    3. Generate candidate ideas
    4. (Optional) Self-critique and refine
    """
    
    def __init__(self, 
                 categorized_papers_path: str,
                 output_dir: str = "generated_ideas"):
        """
        Initialize the generator.
        
        Args:
            categorized_papers_path: Path to categorized_papers_v2.json
            output_dir: Directory to save generated ideas
        """
        self.categorized_papers_path = categorized_papers_path
        self.output_dir = output_dir
        self.categories: Dict[str, List[Paper]] = {}
        
        os.makedirs(output_dir, exist_ok=True)
        self._load_papers()
    
    def _load_papers(self):
        """Load and parse the categorized papers."""
        logger.info(f"Loading papers from {self.categorized_papers_path}")
        
        with open(self.categorized_papers_path, 'r') as f:
            data = json.load(f)
        
        for category, papers in data.items():
            self.categories[category] = [
                Paper(
                    id=p['id'],
                    title=p['title'],
                    conference=p['conference'],
                    keywords=p.get('keywords', []),
                    abstract=p.get('abstract')
                )
                for p in papers
            ]
        
        logger.info(f"Loaded {len(self.categories)} categories")
        for cat, papers in self.categories.items():
            logger.info(f"  {cat}: {len(papers)} papers")
    
    def list_categories(self) -> List[str]:
        """Return list of available categories."""
        return list(self.categories.keys())
    
    def get_category_papers(self, category: str) -> List[Paper]:
        """Get papers for a specific category."""
        if category not in self.categories:
            raise ValueError(f"Category '{category}' not found. Available: {self.list_categories()}")
        return self.categories[category]
    
    def extract_domain_context(self, category: str, max_papers: int = 50) -> Dict:
        """
        Extract context from papers in a domain for idea generation.
        
        Returns a dictionary with:
        - common_keywords: Most frequent keywords
        - paper_summaries: Brief summaries of key papers
        - potential_themes: Identified research themes
        """
        papers = self.get_category_papers(category)[:max_papers]
        
        # Count keyword frequencies
        keyword_counts = {}
        for paper in papers:
            for kw in paper.keywords:
                # Handle both single keywords and semicolon-separated lists
                for k in kw.split(';'):
                    k = k.strip().lower()
                    if k:
                        keyword_counts[k] = keyword_counts.get(k, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: -x[1])
        top_keywords = sorted_keywords[:30]
        
        # Create paper summaries
        paper_summaries = [
            {
                'title': p.title,
                'keywords': p.keywords[:5] if p.keywords else [],
                'conference': p.conference
            }
            for p in papers[:20]  # Top 20 papers for context
        ]
        
        return {
            'category': category,
            'num_papers': len(papers),
            'top_keywords': top_keywords,
            'paper_summaries': paper_summaries,
        }
    
    def generate_prompt_for_ideas(self, domain_context: Dict, num_ideas: int = 5) -> str:
        """
        Generate a prompt for LLM-based idea generation.
        
        This creates a structured prompt that can be sent to an LLM API.
        """
        category = domain_context['category']
        keywords = [kw for kw, _ in domain_context['top_keywords'][:20]]
        papers = domain_context['paper_summaries']
        
        prompt = f"""You are a research scientist generating novel research ideas in the domain of "{category}".

Based on the following context from recent papers in this field, generate {num_ideas} novel research ideas.

## Domain Context

**Category**: {category}
**Number of papers analyzed**: {domain_context['num_papers']}

**Key topics and methods in this field**:
{', '.join(keywords)}

**Recent papers in this area**:
"""
        for i, p in enumerate(papers[:15], 1):
            kws = ', '.join(p['keywords'][:3]) if p['keywords'] else 'N/A'
            prompt += f"{i}. \"{p['title']}\" ({p['conference']}) - Keywords: {kws}\n"
        
        prompt += f"""

## Task

Generate {num_ideas} novel research ideas that:
1. Build on the existing work in this field
2. Address potential gaps or limitations
3. Combine existing methods in new ways
4. Are specific and actionable (not vague)

For each idea, provide:
- **Title**: A concise title for the research idea
- **Description**: 2-3 sentences describing the core idea
- **Motivation**: Why this is interesting/important
- **Related papers**: Which of the above papers relate to this idea

Format each idea as:
---
### Idea [N]: [Title]
**Description**: [description]
**Motivation**: [motivation]  
**Related papers**: [list paper numbers]
---
"""
        return prompt
    
    def generate_critique_prompt(self, idea_text: str, domain_context: Dict) -> str:
        """Generate a prompt for self-critique of an idea."""
        prompt = f"""You are a critical reviewer evaluating a research idea in the domain of "{domain_context['category']}".

## Research Idea
{idea_text}

## Evaluation Criteria
Please evaluate this idea on:
1. **Novelty**: Is this genuinely new, or does it already exist?
2. **Feasibility**: Can this be realistically implemented?
3. **Significance**: Would this be impactful if successful?
4. **Clarity**: Is the idea well-defined and specific?

## Task
Provide:
1. A brief critique (2-3 sentences) highlighting strengths and weaknesses
2. A confidence score from 1-10 (10 = highly promising)
3. One suggestion to improve the idea

Format:
**Critique**: [your critique]
**Score**: [1-10]
**Improvement**: [suggestion]
"""
        return prompt
    
    def save_ideas(self, ideas: List[ResearchIdea], category: str):
        """Save generated ideas to a JSON file."""
        filename = f"{category.replace(' ', '_').replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump([asdict(idea) for idea in ideas], f, indent=2)
        
        logger.info(f"Saved {len(ideas)} ideas to {filepath}")
        return filepath
    
    def run_generation(self, 
                       category: str, 
                       num_ideas: int = 5,
                       include_critique: bool = False) -> Dict:
        """
        Run the full generation pipeline for a category.
        
        Returns a dictionary with:
        - domain_context: Extracted context
        - generation_prompt: The prompt for idea generation
        - critique_prompt: (if include_critique) Template for critique
        
        Note: This doesn't call an LLM - it prepares the prompts.
        The actual LLM calls should be done by the caller.
        """
        logger.info(f"Running generation for category: {category}")
        
        # Extract domain context
        domain_context = self.extract_domain_context(category)
        
        # Generate the main prompt
        generation_prompt = self.generate_prompt_for_ideas(domain_context, num_ideas)
        
        result = {
            'category': category,
            'domain_context': domain_context,
            'generation_prompt': generation_prompt,
            'num_ideas_requested': num_ideas,
        }
        
        if include_critique:
            # Placeholder for critique - actual ideas would be inserted
            result['critique_prompt_template'] = self.generate_critique_prompt(
                "[INSERT IDEA HERE]", domain_context
            )
        
        return result


def main():
    """Demo the single-domain generator."""
    # Path to categorized papers
    papers_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'literature_review', 'categorized_papers_v2.json'
    )
    
    if not os.path.exists(papers_path):
        print(f"Error: {papers_path} not found")
        return
    
    # Initialize generator
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'generated_ideas')
    generator = SingleDomainGenerator(papers_path, output_dir)
    
    # List available categories
    print("\n=== Available Categories ===")
    categories = generator.list_categories()
    for cat in sorted(categories):
        papers = generator.get_category_papers(cat)
        print(f"  {cat}: {len(papers)} papers")
    
    # Demo: Generate for NLP - LLMs & Foundation Models
    demo_category = "NLP - LLMs & Foundation Models"
    if demo_category in categories:
        print(f"\n=== Generating ideas for: {demo_category} ===")
        result = generator.run_generation(demo_category, num_ideas=5, include_critique=True)
        
        # Save the prompt for review
        prompt_file = os.path.join(output_dir, f"prompt_{demo_category.replace(' ', '_')}.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(prompt_file, 'w') as f:
            f.write(result['generation_prompt'])
        print(f"\nGeneration prompt saved to: {prompt_file}")
        
        # Print some stats
        ctx = result['domain_context']
        print(f"\nDomain context:")
        print(f"  Papers analyzed: {ctx['num_papers']}")
        print(f"  Top keywords: {[k for k, _ in ctx['top_keywords'][:10]]}")


if __name__ == "__main__":
    main()

