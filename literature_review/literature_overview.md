# Literature Review: LLM-based Hypothesis/Idea Generation

## Categories

### 1. Hypothesis/Idea Generation Methods

| Paper | Year | Key Contribution | Notes |
|-------|------|------------------|-------|
| MOOSE-Chem (arXiv:2410.07076) | 2024 | Decomposes hypothesis discovery into: retrieving inspirations, composing hypotheses, ranking | ICLR 2025 accepted, chemistry focused, 51 paper benchmark |
| MOOSE-Chem3 (arXiv:2505.17873) | 2025 | Extends MOOSE-Chem with experiment-guided hypothesis ranking | Feedback loops seem powerful |
| AI-Augmented Brainwriting (arXiv:2402.14978) | 2024 | LLMs for both divergent (generation) and convergent (evaluation) stages | Human-AI collaboration model |

### 2. Benchmarks for Evaluation

| Paper | Year | Key Contribution | Notes |
|-------|------|------------------|-------|
| AI Idea Bench 2025 (arXiv:2504.14191) | 2025 | 3,495 AI papers with inspired works for evaluation | Useful for measuring if generated ideas align with published work |
| HypoBench (arXiv:2504.11524) | 2025 | 7 real-world + 5 synthetic tasks, 194 datasets | Best models only recover 38.8% of ground truth hypotheses |

### 3. Cross-Domain / Analogical Reasoning

| Paper | Year | Key Contribution | Notes |
|-------|------|------------------|-------|
| Fluid Transformers and Creative Analogies (arXiv:2302.12832) | 2023 | Tests LLMs' ability to generate cross-domain analogies | Many analogies judged helpful |
| Augmenting Scientific Creativity with Retrieval (arXiv:2206.01328) | 2022 | Retrieval of similar content from different domains | Cross-domain exposure tool |
| Radical Concept Generation (MDPI) | 2022 | Cross-domain knowledge from patents, "technological distance" metric | Framework for structured cross-domain search |

### 4. Literature Mining / Surveys

| Paper | Year | Key Contribution | Notes |
|-------|------|------------------|-------|
| Systematic Literature Review about Idea Mining (arXiv:2202.12826) | 2022 | Surveys 71 papers on idea mining techniques | Text mining, IR, NLP, deep learning |
| AI in Literature Reviews (arXiv:2402.08565) | 2024 | 21 tools surveyed, 11 using LLMs | Semi-automation of lit review |

---

## Key Dimensions for Categorization

When reviewing papers, capture:
- **Method of Idea Gen**: Human-only, AI-only, human+AI collaboration
- **Domain(s)**: Single domain vs. cross-domain; specifics (chemistry, ML, NLP, etc.)
- **Data Sources**: Papers, patents, social media, etc.
- **Evaluation Metrics**: Novelty, usefulness, plausibility, expert judgment
- **Techniques**: NLP, analogical reasoning, retrieval, LLM prompting, graph methods
- **Limitations**: Bias, domain-knowledge requirements, hallucination, evaluation challenges

---

## Papers We've Discussed

Papers mentioned by Aldous:
1. https://arxiv.org/abs/2410.07076 - MOOSE-Chem
2. https://arxiv.org/abs/2504.14191 - AI Idea Bench 2025
3. https://arxiv.org/abs/2504.11524 - HypoBench

---

## TODO
- [ ] Find more recent 2025 papers on cross-domain idea generation
- [ ] Look into evaluation frameworks for "valuable ideas"
- [ ] Investigate how MOOSE-Chem could be adapted for non-chemistry domains
- [ ] Build cross-domain retrieval using our paper scraper

