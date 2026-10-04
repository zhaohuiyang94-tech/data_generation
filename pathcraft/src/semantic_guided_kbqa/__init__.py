"""Semantic-guided Freebase KBQA pipeline."""

__version__ = "0.3.9-webqsp-llamafactory-guarded-review"

from .config import AppConfig
from .pipeline import SemanticGuidedPipeline

__all__ = ["AppConfig", "SemanticGuidedPipeline"]
