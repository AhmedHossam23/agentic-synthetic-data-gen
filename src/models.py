"""Data models for reviews and quality metrics."""

from typing import Optional, Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Review(BaseModel):
    """A single review with metadata."""
    id: str
    text: str
    rating: int = Field(ge=1, le=5)
    persona: str
    model_used: str
    tool_name: Optional[str] = None
    category: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.now)
    
    # Quality scores
    quality_scores: Optional[Dict[str, float]] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None


class QualityMetrics(BaseModel):
    """Quality metrics for a review."""
    diversity_score: float = Field(ge=0.0, le=1.0)
    bias_score: float = Field(ge=0.0, le=1.0)
    realism_score: float = Field(ge=0.0, le=1.0)
    overall_quality: float = Field(ge=0.0, le=1.0)
    
    # Detailed metrics
    vocabulary_overlap: Optional[float] = None
    semantic_similarity: Optional[float] = None
    sentiment_score: Optional[float] = None
    readability_score: Optional[float] = None
    domain_term_ratio: Optional[float] = None


class GenerationStats(BaseModel):
    """Statistics for generation process."""
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    rejection_reasons: Dict[str, int] = Field(default_factory=dict)
    
    # Per model stats
    model_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Timing
    total_time_seconds: float = 0.0
    avg_time_per_review: float = 0.0
    
    # Quality distribution
    quality_distribution: Dict[str, int] = Field(default_factory=dict)


class ComparisonMetrics(BaseModel):
    """Metrics comparing synthetic vs real reviews."""
    synthetic_stats: Dict[str, Any]
    real_stats: Dict[str, Any]
    similarity_metrics: Dict[str, float]
    differences: List[str]
