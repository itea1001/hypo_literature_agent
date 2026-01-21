"""
Cross-domain idea merger.

This module takes ideas from different domains and attempts to merge/combine them
to create novel cross-domain research ideas.

This is Phase 2 of the idea generation pipeline:
1. Phase 1: Generate single-domain ideas (single_domain_generator.py)
2. Phase 2: Merge ideas across domains (this file)

The key insight: single-domain ideas tend to be more grounded/sound,
while cross-domain merging adds novelty.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MergedIdea:
    """Represents a merged cross-domain idea."""
    id: str
    title: str
    description: str
    source_domain_a: str
    source_domain_b: str
    source_idea_a: str  # Brief description of source idea from domain A
    source_idea_b: str  # Brief description of source idea from domain B
    merge_rationale: str  # Why these ideas can be combined
    generated_at: str
    novelty_score: Optional[float] = None
    feasibility_score: Optional[float] = None


class IdeaMerger:
    """
    Merges ideas from different domains to create novel cross-domain research ideas.
    
    Strategy:
    1. Take generated ideas from two different domains
    2. Identify structural similarities or complementary aspects
    3. Generate merged ideas that combine elements from both
    4. Evaluate for novelty (should be higher) and feasibility (might be lower)
    """
    
    def __init__(self, output_dir: str = "generated_ideas"):
        """Initialize the merger."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_generated_ideas(self, ideas_file: str) -> Dict:
        """Load previously generated ideas from a JSON file."""
        with open(ideas_file, 'r') as f:
            return json.load(f)
    
    def find_domain_pairs(self, 
                          categories: List[str], 
                          distance_metric: str = "keyword_overlap") -> List[Tuple[str, str]]:
        """
        Find promising domain pairs for cross-pollination.
        
        The idea is to find domains that are:
        - Different enough to provide novelty
        - Similar enough in structure to allow meaningful combination
        
        Args:
            categories: List of category names
            distance_metric: How to measure domain distance
            
        Returns:
            List of (domain_a, domain_b) pairs
        """
        # For now, just return some pre-defined interesting pairs
        # TODO: Implement automatic pair finding based on embeddings or keyword overlap
        
        interesting_pairs = [
            # Method transfer pairs
            ("NLP - LLMs & Foundation Models", "Application - Biology & Medicine"),
            ("NLP - Prompting & In-Context Learning", "Application - Chemistry & Materials"),
            ("Theory - Optimization", "Application - Biology & Medicine"),
            ("Generative - Diffusion Models", "Application - Biology & Medicine"),
            
            # Cross-modal pairs
            ("NLP - Generation", "CV - Image Generation"),
            ("NLP - Understanding", "CV - Video Understanding"),
            
            # Theory + Application pairs  
            ("Theory - Generalization", "NLP - LLMs & Foundation Models"),
            ("Theory - Learning Theory", "RL - Policy Learning"),
            
            # Safety across domains
            ("Safety - Alignment & Safety", "Application - Biology & Medicine"),
            ("Safety - Fairness", "NLP - LLMs & Foundation Models"),
        ]
        
        # Filter to only include pairs where both domains exist in categories
        valid_pairs = [
            (a, b) for a, b in interesting_pairs 
            if a in categories and b in categories
        ]
        
        return valid_pairs
    
    def generate_merge_prompt(self,
                              domain_a: str,
                              domain_b: str,
                              ideas_a: str,
                              ideas_b: str,
                              num_merged: int = 3) -> str:
        """
        Generate a prompt for merging ideas from two domains.
        
        Args:
            domain_a: First domain name
            domain_b: Second domain name
            ideas_a: Generated ideas text from domain A
            ideas_b: Generated ideas text from domain B
            num_merged: Number of merged ideas to generate
        """
        prompt = f"""You are a creative research scientist exploring cross-domain research opportunities.

You have research ideas from two different domains. Your task is to find novel ways to combine or merge insights from both domains.

## Domain A: {domain_a}
### Ideas from Domain A:
{ideas_a}

---

## Domain B: {domain_b}
### Ideas from Domain B:
{ideas_b}

---

## Task

Generate {num_merged} novel research ideas that MERGE concepts from both domains. 

Focus on:
1. **Method Transfer**: Can a method from Domain A be applied to problems in Domain B?
2. **Analogical Reasoning**: Are there structural similarities that suggest new approaches?
3. **Complementary Combination**: Can elements from both domains combine to solve new problems?

For each merged idea, provide:
- **Title**: A concise title
- **Description**: 2-3 sentences describing the merged idea
- **From Domain A**: Which aspect/idea from Domain A contributes
- **From Domain B**: Which aspect/idea from Domain B contributes
- **Merge Rationale**: Why this combination makes sense
- **Expected Novelty**: Why this is novel (1 sentence)
- **Potential Challenges**: What might make this difficult (1 sentence)

Format each idea as:
---
### Merged Idea [N]: [Title]
**Description**: [description]
**From Domain A ({domain_a})**: [contribution from A]
**From Domain B ({domain_b})**: [contribution from B]
**Merge Rationale**: [why this combination works]
**Expected Novelty**: [novelty argument]
**Potential Challenges**: [challenges]
---
"""
        return prompt
    
    def generate_evaluation_prompt(self, merged_idea: str) -> str:
        """Generate a prompt to evaluate a merged idea."""
        prompt = f"""You are evaluating a cross-domain research idea for quality.

## Merged Idea
{merged_idea}

## Evaluation Criteria

Rate the idea on these dimensions (1-10 scale):

1. **Novelty**: How new/unique is this combination? (Higher is better)
2. **Feasibility**: How realistic is implementation? (Consider both domains' constraints)
3. **Significance**: If successful, how impactful would this be?
4. **Coherence**: Does the merge make logical sense, or is it forced?

Provide:
- Scores for each criterion
- A brief overall assessment (2-3 sentences)
- One suggestion to improve the idea

Format:
**Novelty**: [score]/10
**Feasibility**: [score]/10
**Significance**: [score]/10
**Coherence**: [score]/10
**Overall Assessment**: [assessment]
**Improvement Suggestion**: [suggestion]
"""
        return prompt
    
    def save_merged_ideas(self, 
                          merged_ideas: List[MergedIdea],
                          domain_a: str,
                          domain_b: str) -> str:
        """Save merged ideas to a file."""
        safe_a = domain_a.replace(' ', '_').replace('/', '-').replace('&', 'and')
        safe_b = domain_b.replace(' ', '_').replace('/', '-').replace('&', 'and')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f"merged_{safe_a}_x_{safe_b}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump([asdict(idea) for idea in merged_ideas], f, indent=2)
        
        logger.info(f"Saved {len(merged_ideas)} merged ideas to {filepath}")
        return filepath


def main():
    """Demo the idea merger."""
    print("\n=== Cross-Domain Idea Merger ===\n")
    print("This module merges ideas from different domains.")
    print("It's Phase 2 of the idea generation pipeline.\n")
    
    # Show interesting pairs
    merger = IdeaMerger()
    
    sample_categories = [
        "NLP - LLMs & Foundation Models",
        "Application - Biology & Medicine", 
        "Theory - Optimization",
        "Generative - Diffusion Models",
        "Safety - Alignment & Safety",
    ]
    
    pairs = merger.find_domain_pairs(sample_categories)
    print("Interesting domain pairs for merging:")
    for a, b in pairs:
        print(f"  - {a} × {b}")
    
    # Show a sample merge prompt
    print("\n=== Sample Merge Prompt ===\n")
    sample_prompt = merger.generate_merge_prompt(
        "NLP - LLMs & Foundation Models",
        "Application - Biology & Medicine",
        "Idea 1: Improved in-context learning for multi-step reasoning...",
        "Idea 1: Protein structure prediction using graph neural networks...",
        num_merged=2
    )
    print(sample_prompt[:1500] + "...\n")


if __name__ == "__main__":
    main()

