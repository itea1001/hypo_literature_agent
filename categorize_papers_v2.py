#!/usr/bin/env python3
"""
Script to categorize papers from NeurIPS, ICML, and ICLR 2024
with fine-grained categories and cross-disciplinary applications
"""
import json
from pathlib import Path
from collections import defaultdict

def load_papers(filepath):
    """Load papers from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def categorize_paper(paper):
    """
    Categorize a paper based on its title, abstract, and keywords.
    Returns a list of categories the paper belongs to.
    """
    text = (paper.get('title', '') + ' ' + 
            paper.get('abstract', '') + ' ' + 
            ' '.join(paper.get('keywords', []))).lower()
    
    categories = []
    
    # Define fine-grained category keywords
    category_keywords = {
        # === NLP Sub-categories ===
        'NLP - LLMs & Foundation Models': ['language model', 'llm', 'gpt', 'bert', 'transformer',
                                           'foundation model', 'pre-training', 'fine-tuning'],
        'NLP - Generation': ['text generation', 'generation', 'summarization', 'dialogue',
                            'conversation', 'chatbot'],
        'NLP - Understanding': ['understanding', 'comprehension', 'reasoning', 'question answering',
                               'qa', 'reading comprehension'],
        'NLP - Prompting & In-Context Learning': ['prompt', 'prompting', 'in-context', 'icl',
                                                  'few-shot', 'chain-of-thought'],
        
        # === Computer Vision Sub-categories ===
        'CV - Image Generation': ['image generation', 'diffusion', 'gan', 'image synthesis',
                                 'text-to-image', 'stable diffusion'],
        'CV - Recognition & Detection': ['object detection', 'recognition', 'classification',
                                        'segmentation', 'tracking'],
        'CV - Video Understanding': ['video', 'temporal', 'action recognition', 'motion'],
        'CV - 3D Vision': ['3d', 'point cloud', 'mesh', 'depth', 'reconstruction'],
        
        # === RL Sub-categories ===
        'RL - Policy Learning': ['policy', 'actor-critic', 'ppo', 'ddpg', 'sac'],
        'RL - Multi-Agent': ['multi-agent', 'marl', 'cooperative', 'competitive'],
        'RL - Offline RL': ['offline', 'batch', 'offline reinforcement'],
        'RL - Exploration': ['exploration', 'curiosity', 'intrinsic motivation'],
        'RL - Meta-RL': ['meta-learning', 'meta-rl', 'adaptation', 'transfer'],
        
        # === ML Theory ===
        'Theory - Optimization': ['convergence', 'gradient descent', 'optimization theory',
                                 'convex', 'non-convex'],
        'Theory - Generalization': ['generalization', 'pac', 'sample complexity', 'vc dimension',
                                   'rademacher'],
        'Theory - Learning Theory': ['learning theory', 'statistical learning', 'regret bound'],
        
        # === Graph & Network Learning ===
        'Graph - GNNs': ['graph neural network', 'gnn', 'message passing', 'graph convolution'],
        'Graph - Node/Link Prediction': ['node classification', 'link prediction', 'node embedding'],
        'Graph - Applications': ['social network', 'knowledge graph', 'molecular graph'],
        
        # === Generative Models ===
        'Generative - Diffusion Models': ['diffusion', 'score-based', 'denoising'],
        'Generative - GANs': ['gan', 'generative adversarial'],
        'Generative - VAEs & Flow': ['vae', 'variational autoencoder', 'normalizing flow'],
        'Generative - Autoregressive': ['autoregressive', 'pixelcnn', 'wavenet'],
        
        # === Robustness & Safety ===
        'Safety - Adversarial Robustness': ['adversarial', 'attack', 'defense', 'perturbation'],
        'Safety - Privacy': ['privacy', 'differential privacy', 'federated', 'private'],
        'Safety - Alignment & Safety': ['alignment', 'safety', 'harmful', 'jailbreak', 'toxicity'],
        'Safety - Fairness': ['fairness', 'bias', 'equitable', 'discrimination'],
        
        # === Interpretability & Explainability ===
        'Interpretability - Mechanistic': ['mechanistic interpretability', 'circuit', 'neuron'],
        'Interpretability - Attribution': ['attribution', 'saliency', 'attention visualization'],
        'Interpretability - Concept Learning': ['concept', 'prototype', 'exemplar'],
        
        # === Cross-Disciplinary Applications ===
        'Application - Biology & Medicine': ['biology', 'protein', 'gene', 'cell', 'disease',
                                            'medical', 'clinical', 'healthcare', 'drug', 'therapy'],
        'Application - Chemistry & Materials': ['chemistry', 'chemical', 'molecular', 'molecule',
                                               'material', 'catalyst', 'compound'],
        'Application - Physics & Engineering': ['physics', 'quantum', 'simulation', 'fluid',
                                               'engineering', 'control', 'robotics'],
        'Application - Neuroscience': ['neuroscience', 'brain', 'neural', 'cognitive', 'fmri'],
        'Application - Social Sciences': ['social', 'economics', 'game theory', 'mechanism design',
                                         'market', 'psychology'],
        'Application - Climate & Environment': ['climate', 'weather', 'environment', 'sustainability',
                                               'carbon', 'energy'],
        'Application - Finance & Economics': ['finance', 'financial', 'trading', 'portfolio',
                                             'economic', 'pricing'],
        
        # === Methodology ===
        'Method - Transfer & Domain Adaptation': ['transfer learning', 'domain adaptation',
                                                  'domain shift', 'distribution shift'],
        'Method - Few-Shot & Zero-Shot': ['few-shot', 'zero-shot', 'one-shot'],
        'Method - Self-Supervised Learning': ['self-supervised', 'contrastive', 'pretext task'],
        'Method - Multi-Task Learning': ['multi-task', 'multitask', 'task learning'],
        'Method - Active Learning': ['active learning', 'query', 'annotation'],
        'Method - Continual Learning': ['continual', 'lifelong', 'catastrophic forgetting'],
        
        # === Systems & Efficiency ===
        'Systems - Efficiency & Compression': ['efficiency', 'compression', 'pruning', 'quantization',
                                              'distillation', 'sparse'],
        'Systems - Distributed Training': ['distributed', 'parallel', 'multi-gpu', 'scalability'],
        'Systems - Hardware': ['hardware', 'accelerator', 'gpu', 'tpu', 'fpga'],
        
        # === Data & Evaluation ===
        'Data - Datasets & Benchmarks': ['benchmark', 'dataset', 'corpus', 'collection'],
        'Data - Data Quality': ['data quality', 'noise', 'labeling', 'annotation'],
        'Data - Synthetic Data': ['synthetic', 'simulation', 'procedural generation'],
        
        # === Multi-Modal ===
        'Multi-Modal - Vision-Language': ['vision-language', 'vqa', 'image captioning', 'clip'],
        'Multi-Modal - Audio-Visual': ['audio', 'speech', 'sound', 'acoustic'],
        'Multi-Modal - Embodied AI': ['embodied', 'navigation', 'manipulation', 'interaction'],
    }
    
    # Check which categories apply
    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            categories.append(category)
    
    # If no category found, mark as "Other"
    if not categories:
        categories = ['Other']
    
    return categories

def main():
    # Load all papers
    data_dir = Path('literature_review/papers/conferences')
    all_papers = []
    
    for conf_file in ['neurips_2024_full.json', 'icml_2024_full.json', 'iclr_2024_full.json']:
        filepath = data_dir / conf_file
        papers = load_papers(filepath)
        all_papers.extend(papers)
        print(f"Loaded {len(papers)} papers from {conf_file}")
    
    print(f"\nTotal papers: {len(all_papers)}")
    
    # Categorize papers
    categorized = defaultdict(list)
    paper_to_categories = {}
    
    for paper in all_papers:
        categories = categorize_paper(paper)
        paper_to_categories[paper['id']] = categories
        
        for category in categories:
            categorized[category].append({
                'id': paper['id'],
                'title': paper['title'],
                'conference': paper['conference'],
                'keywords': paper.get('keywords', []),
            })
    
    # Print statistics by major category
    print("\n=== Category Statistics (Fine-Grained) ===")
    
    # Group by major category
    major_categories = defaultdict(list)
    for category in categorized.keys():
        if ' - ' in category:
            major = category.split(' - ')[0]
            major_categories[major].append(category)
        else:
            major_categories['Other'].append(category)
    
    for major in sorted(major_categories.keys()):
        print(f"\n{major}:")
        subcats = sorted(major_categories[major], 
                        key=lambda x: len(categorized[x]), reverse=True)
        for subcat in subcats:
            print(f"  {subcat}: {len(categorized[subcat])} papers")
    
    # Save categorized results
    output_file = 'literature_review/categorized_papers_v2.json'
    with open(output_file, 'w') as f:
        json.dump(dict(categorized), f, indent=2)
    print(f"\nSaved categorized results to {output_file}")
    
    # Save paper-to-category mapping
    mapping_file = 'literature_review/paper_category_mapping_v2.json'
    with open(mapping_file, 'w') as f:
        json.dump(paper_to_categories, f, indent=2)
    print(f"Saved paper-to-category mapping to {mapping_file}")

if __name__ == '__main__':
    main()

