"""
Idea Evaluator using LiveIdeaBench-style LLM critique.

Evaluates research ideas on originality, feasibility, and clarity
using GPT-4o or similar models as judges.

Adapted from LiveIdeaBench (https://github.com/x66ccff/liveideabench)
"""

import json
import re
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Critic prompt adapted from LiveIdeaBench
CRITIC_PROMPT = """You are an extremely demanding scientific reviewer with the highest critical standards, like those at Nature or Science. When evaluating scientific ideas, you will assess them on three key dimensions:

1. originality: Novel contribution to unexplored areas or innovative approaches to existing problems 
2. feasibility: Technical implementation and practicality
3. clarity: How well-articulated and easy to understand the idea is

Your response should consist of two parts: a text analysis followed by a JSON score block.

First, provide your brief analysis (less than 100 words) of the idea. Then, for each dimension, provide a score from 1 to 10 where 1-3 = poor, 4-6 = average, 7-10 = excellent.

For example:
```json
{
    "originality": <score_1_to_10>,
    "feasibility": <score_1_to_10>,
    "clarity": <score_1_to_10>
}```"""


def parse_scores(raw_text: str) -> Dict[str, Any]:
    """Parse critique text, extract reasoning and scores.
    
    Adapted from LiveIdeaBench's parse_critique function.
    """
    # Try method 1: Extract content between ```json and ```
    json_pattern1 = r'```json(.*?)```'
    match = re.search(json_pattern1, raw_text, re.DOTALL)
    
    if not match:
        # Try method 2: Extract content between ``` and ```
        json_pattern2 = r'```(.*?)```'
        match = re.search(json_pattern2, raw_text, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
        try:
            rating_dict = json.loads(json_str)
            return {
                "is_valid": True,
                "scores": {
                    "originality": rating_dict.get("originality"),
                    "feasibility": rating_dict.get("feasibility"),
                    "clarity": rating_dict.get("clarity")
                },
                "reasoning": raw_text.split("```")[0].strip() if "```" in raw_text else raw_text
            }
        except json.JSONDecodeError:
            pass
    
    # Try method 3: Directly match content within curly braces {}
    braces_pattern = r'\{(.*?)\}'
    match = re.search(braces_pattern, raw_text, re.DOTALL)
    
    if match:
        json_str = '{' + match.group(1) + '}'
        try:
            rating_dict = json.loads(json_str)
            return {
                "is_valid": True,
                "scores": {
                    "originality": rating_dict.get("originality"),
                    "feasibility": rating_dict.get("feasibility"),
                    "clarity": rating_dict.get("clarity")
                },
                "reasoning": raw_text
            }
        except json.JSONDecodeError:
            pass
    
    # Try method 4: Directly match integers after keywords
    results = {}
    for key in ["originality", "feasibility", "clarity"]:
        pattern = r'"{0}"\s*:\s*(\d+)'.format(key)
        match = re.search(pattern, raw_text)
        if match:
            results[key] = int(match.group(1))
        else:
            pattern = r'{0}\s*:\s*(\d+)'.format(key)
            match = re.search(pattern, raw_text)
            if match:
                results[key] = int(match.group(1))
    
    if results:
        return {
            "is_valid": len(results) == 3,
            "scores": results,
            "reasoning": raw_text
        }
    
    return {"is_valid": False, "reasoning": raw_text}


class IdeaEvaluator:
    """Evaluates research ideas using LLM-as-judge approach."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the evaluator.
        
        Args:
            api_key: OpenAI API key. If None, reads from env or file.
            model: Model to use for evaluation (default: gpt-4o-mini)
        """
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        # Get API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            api_key_path = os.path.join(os.path.dirname(__file__), "..", "openai_api_key.txt")
            if os.path.exists(api_key_path):
                with open(api_key_path, 'r') as f:
                    api_key = f.read().strip()
        
        if not api_key:
            raise ValueError("No OpenAI API key found")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def evaluate_idea(self, idea_text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Evaluate a single research idea.
        
        Args:
            idea_text: The research idea to evaluate
            max_retries: Maximum retry attempts for failed parses
            
        Returns:
            Dict with scores and reasoning
        """
        prompt = f"Please evaluate the following scientific research idea:\n\n{idea_text}"
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": CRITIC_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                
                raw_critique = response.choices[0].message.content
                parsed = parse_scores(raw_critique)
                
                if parsed.get("is_valid"):
                    return {
                        "scores": parsed["scores"],
                        "reasoning": parsed.get("reasoning", ""),
                        "raw_response": raw_critique,
                        "model": self.model,
                        "attempts": attempt + 1
                    }
                else:
                    logger.warning(f"Parse failed (attempt {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}): {e}")
        
        # All retries failed
        return {
            "scores": None,
            "reasoning": "Failed to parse evaluation",
            "raw_response": raw_critique if 'raw_critique' in locals() else None,
            "model": self.model,
            "error": "Parsing failed after retries"
        }
    
    def evaluate_ideas_file(self, input_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        Evaluate all ideas in a JSON file.
        
        Args:
            input_path: Path to JSON file with generated ideas
            output_path: Path to save evaluation results
            
        Returns:
            List of evaluation results
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        results = []
        
        # Handle both structured ideas and raw text format
        if "ideas" in data:
            ideas = data.get("ideas", [])
        elif "generated_ideas_raw" in data:
            ideas = self._parse_raw_ideas(data["generated_ideas_raw"])
        else:
            raise ValueError("No ideas found in file")
        
        logger.info(f"Evaluating {len(ideas)} ideas from {input_path}")
        
        for i, idea in enumerate(ideas):
            # Format idea text for evaluation
            idea_text = self._format_idea_for_eval(idea)
            
            logger.info(f"Evaluating idea {i+1}/{len(ideas)}: {idea.get('title', 'Untitled')[:50]}...")
            
            eval_result = self.evaluate_idea(idea_text)
            eval_result["idea_index"] = i
            eval_result["idea_title"] = idea.get("title", "Untitled")
            results.append(eval_result)
            
            if eval_result.get("scores"):
                scores = eval_result["scores"]
                logger.info(f"  Scores: originality={scores.get('originality')}, "
                          f"feasibility={scores.get('feasibility')}, clarity={scores.get('clarity')}")
            else:
                logger.warning(f"  Failed to get scores")
        
        # Calculate summary statistics
        valid_results = [r for r in results if r.get("scores")]
        if valid_results:
            avg_scores = {
                "originality": sum(r["scores"]["originality"] for r in valid_results) / len(valid_results),
                "feasibility": sum(r["scores"]["feasibility"] for r in valid_results) / len(valid_results),
                "clarity": sum(r["scores"]["clarity"] for r in valid_results) / len(valid_results)
            }
            logger.info(f"\nAverage scores across {len(valid_results)} ideas:")
            logger.info(f"  Originality: {avg_scores['originality']:.2f}")
            logger.info(f"  Feasibility: {avg_scores['feasibility']:.2f}")
            logger.info(f"  Clarity: {avg_scores['clarity']:.2f}")
        
        # Save results
        output_data = {
            "source_file": input_path,
            "eval_model": self.model,
            "timestamp": datetime.now().isoformat(),
            "num_ideas": len(ideas),
            "num_evaluated": len(valid_results),
            "average_scores": avg_scores if valid_results else None,
            "evaluations": results
        }
        
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_evaluated.json"
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return results
    
    def _parse_raw_ideas(self, raw_text: str) -> List[Dict]:
        """Parse raw markdown-formatted ideas into structured format."""
        ideas = []
        
        # Split by idea markers
        idea_blocks = re.split(r'---\s*\n### Idea \d+:', raw_text)
        
        for block in idea_blocks:
            if not block.strip():
                continue
            
            idea = {}
            
            # Extract title (first line after the split)
            title_match = re.match(r'\s*([^\n]+)', block)
            if title_match:
                idea['title'] = title_match.group(1).strip()
            
            # Extract each section
            sections = {
                'problem_statement': r'\*\*Problem Statement\*\*:\s*(.+?)(?=\*\*|$)',
                'proposed_method': r'\*\*Proposed Method\*\*:\s*(.+?)(?=\*\*|$)',
                'datasets_benchmarks': r'\*\*Datasets & Benchmarks\*\*:\s*(.+?)(?=\*\*|$)',
                'baselines': r'\*\*Baselines\*\*:\s*(.+?)(?=\*\*|$)',
                'expected_results': r'\*\*Expected Results\*\*:\s*(.+?)(?=\*\*|$)',
                'key_risks': r'\*\*Key Risks\*\*:\s*(.+?)(?=\*\*|$)'
            }
            
            for key, pattern in sections.items():
                match = re.search(pattern, block, re.DOTALL)
                if match:
                    idea[key] = match.group(1).strip()
            
            if idea.get('title') or idea.get('problem_statement'):
                ideas.append(idea)
        
        return ideas
    
    def _format_idea_for_eval(self, idea: Dict) -> str:
        """Format an idea dict into text for evaluation."""
        parts = []
        
        if idea.get("title"):
            parts.append(f"**Title**: {idea['title']}")
        
        if idea.get("problem_statement"):
            parts.append(f"**Problem Statement**: {idea['problem_statement']}")
        
        if idea.get("proposed_method"):
            parts.append(f"**Proposed Method**: {idea['proposed_method']}")
        
        if idea.get("datasets_benchmarks"):
            parts.append(f"**Datasets & Benchmarks**: {idea['datasets_benchmarks']}")
        
        if idea.get("baselines"):
            parts.append(f"**Baselines**: {idea['baselines']}")
        
        if idea.get("expected_results"):
            parts.append(f"**Expected Results**: {idea['expected_results']}")
        
        if idea.get("key_risks"):
            parts.append(f"**Key Risks**: {idea['key_risks']}")
        
        # Fallback for simpler formats
        if not parts and idea.get("description"):
            parts.append(idea["description"])
        
        return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Evaluate research ideas using LLM-as-judge")
    parser.add_argument("--input", "-i", required=True, help="Path to ideas JSON file")
    parser.add_argument("--output", "-o", help="Path to save results (default: <input>_evaluated.json)")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="Evaluation model (default: gpt-4o-mini)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    evaluator = IdeaEvaluator(api_key=args.api_key, model=args.model)
    evaluator.evaluate_ideas_file(args.input, args.output)


if __name__ == "__main__":
    main()

