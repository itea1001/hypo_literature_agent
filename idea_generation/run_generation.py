"""
Run idea generation with LLM.

This script runs the single-domain idea generation pipeline,
calling an LLM to generate and optionally critique ideas.

Usage:
    python run_generation.py --category "NLP - LLMs & Foundation Models" --num-ideas 5
    python run_generation.py --category "Application - Biology & Medicine" --with-critique
    python run_generation.py --list-categories
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from typing import Optional

# Try to import openai
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from single_domain_generator import SingleDomainGenerator, ResearchIdea

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMIdeaGenerator:
    """
    Wrapper that combines SingleDomainGenerator with actual LLM calls.
    """
    
    def __init__(self, 
                 papers_path: str,
                 output_dir: str,
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None):
        """
        Initialize the LLM-powered generator.
        
        Args:
            papers_path: Path to categorized_papers_v2.json
            output_dir: Directory to save generated ideas
            model: OpenAI model to use
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.generator = SingleDomainGenerator(papers_path, output_dir)
        self.model = model
        self.output_dir = output_dir
        
        if HAS_OPENAI:
            if api_key:
                openai.api_key = api_key
            elif os.environ.get('OPENAI_API_KEY'):
                openai.api_key = os.environ['OPENAI_API_KEY']
            else:
                logger.warning("No OpenAI API key found. Set OPENAI_API_KEY or pass api_key.")
    
    def call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call the LLM with a prompt."""
        if not HAS_OPENAI:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        
        if not openai.api_key:
            raise RuntimeError("No OpenAI API key configured")
        
        logger.info(f"Calling {self.model}...")
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a creative research scientist who generates novel, specific, and actionable research ideas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.8  # Slightly higher for creativity
        )
        
        return response.choices[0].message.content
    
    def generate_ideas(self, 
                       category: str, 
                       num_ideas: int = 5,
                       with_critique: bool = False) -> dict:
        """
        Generate ideas for a category using LLM.
        
        Returns:
            Dictionary with generation results including raw LLM output
        """
        # Get generation context and prompt
        result = self.generator.run_generation(category, num_ideas, include_critique=with_critique)
        
        # Call LLM to generate ideas
        logger.info(f"Generating {num_ideas} ideas for '{category}'...")
        ideas_text = self.call_llm(result['generation_prompt'])
        result['generated_ideas_raw'] = ideas_text
        
        # Optionally critique
        if with_critique:
            logger.info("Running self-critique on generated ideas...")
            critique_prompt = f"""You are reviewing the following research ideas for quality and novelty.

## Generated Ideas
{ideas_text}

## Task
For each idea, provide:
1. A brief assessment of its novelty and feasibility
2. A score from 1-10 (10 = highly promising)
3. One specific improvement suggestion

Be critical but constructive.
"""
            critique_text = self.call_llm(critique_prompt)
            result['critique'] = critique_text
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_category = category.replace(' ', '_').replace('/', '-').replace('&', 'and')
        output_file = os.path.join(self.output_dir, f"ideas_{safe_category}_{timestamp}.json")
        
        # Save serializable version
        save_result = {
            'category': category,
            'num_ideas_requested': num_ideas,
            'model': self.model,
            'timestamp': timestamp,
            'domain_context': {
                'num_papers': result['domain_context']['num_papers'],
                'top_keywords': result['domain_context']['top_keywords'][:15],
            },
            'generated_ideas_raw': ideas_text,
        }
        if with_critique:
            save_result['critique'] = critique_text
        
        with open(output_file, 'w') as f:
            json.dump(save_result, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        result['output_file'] = output_file
        
        return result
    
    def generate_prompt_only(self, category: str, num_ideas: int = 5) -> str:
        """Generate just the prompt without calling LLM (for testing/review)."""
        result = self.generator.run_generation(category, num_ideas)
        return result['generation_prompt']


def main():
    parser = argparse.ArgumentParser(description='Generate research ideas using LLM')
    parser.add_argument('--category', type=str, help='Category to generate ideas for')
    parser.add_argument('--num-ideas', type=int, default=5, help='Number of ideas to generate')
    parser.add_argument('--with-critique', action='store_true', help='Include self-critique step')
    parser.add_argument('--list-categories', action='store_true', help='List available categories')
    parser.add_argument('--prompt-only', action='store_true', help='Just output the prompt, don\'t call LLM')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    papers_path = os.path.join(script_dir, '..', 'literature_review', 'categorized_papers_v2.json')
    output_dir = os.path.join(script_dir, '..', 'generated_ideas')
    
    if not os.path.exists(papers_path):
        print(f"Error: Categorized papers not found at {papers_path}")
        sys.exit(1)
    
    # Initialize
    llm_gen = LLMIdeaGenerator(papers_path, output_dir, model=args.model)
    
    if args.list_categories:
        print("\n=== Available Categories ===\n")
        categories = llm_gen.generator.list_categories()
        for cat in sorted(categories):
            papers = llm_gen.generator.get_category_papers(cat)
            print(f"  {cat}: {len(papers)} papers")
        return
    
    if not args.category:
        print("Error: --category is required (or use --list-categories)")
        sys.exit(1)
    
    if args.prompt_only:
        prompt = llm_gen.generate_prompt_only(args.category, args.num_ideas)
        print("\n=== Generation Prompt ===\n")
        print(prompt)
        return
    
    # Run full generation
    try:
        result = llm_gen.generate_ideas(
            args.category, 
            args.num_ideas, 
            with_critique=args.with_critique
        )
        
        print("\n" + "="*60)
        print(f"Generated Ideas for: {args.category}")
        print("="*60 + "\n")
        print(result['generated_ideas_raw'])
        
        if args.with_critique:
            print("\n" + "="*60)
            print("Self-Critique")
            print("="*60 + "\n")
            print(result['critique'])
        
        print(f"\n\nResults saved to: {result['output_file']}")
        
    except Exception as e:
        logger.error(f"Error generating ideas: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

