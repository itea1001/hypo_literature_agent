"""
Idea generation framework for scientific hypothesis discovery.

Modules:
- single_domain_generator: Generate ideas within a single research domain
- idea_merger: (TODO) Merge ideas across domains for cross-pollination
"""

from .single_domain_generator import SingleDomainGenerator, Paper, ResearchIdea

__all__ = ['SingleDomainGenerator', 'Paper', 'ResearchIdea']

