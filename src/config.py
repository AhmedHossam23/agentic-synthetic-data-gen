"""Configuration management for the synthetic data generator."""

import yaml
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    name: str
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    weight: float = 0.5


class PersonaConfig(BaseModel):
    """Configuration for a persona."""
    name: str
    role: str
    company_size: str
    experience_years: int
    focus_areas: List[str]
    language_style: str
    rating_distribution: Dict[int, float]


class ReviewCharacteristics(BaseModel):
    """Review generation characteristics."""
    min_length: int = 50
    max_length: int = 800
    min_sentences: int = 3
    max_sentences: int = 15
    include_pros: bool = True
    include_cons: bool = True
    include_use_case: bool = True
    include_rating_justification: bool = True


class DiversityConfig(BaseModel):
    """Diversity metrics configuration."""
    min_vocabulary_overlap: float = 0.3
    max_vocabulary_overlap: float = 0.7
    min_semantic_similarity: float = 0.2
    max_semantic_similarity: float = 0.85


class BiasDetectionConfig(BaseModel):
    """Bias detection configuration."""
    max_sentiment_skew: float = 0.3
    min_rating_variance: float = 0.5
    check_unrealistic_patterns: bool = True


class RealismConfig(BaseModel):
    """Realism validation configuration."""
    min_readability_score: float = 30
    max_readability_score: float = 80
    check_domain_terms: bool = True
    min_domain_term_ratio: float = 0.05


class RejectionThresholds(BaseModel):
    """Quality rejection thresholds."""
    diversity_score: float = 0.3
    bias_score: float = 0.7
    realism_score: float = 0.4
    overall_quality: float = 0.5


class QualityGuardrails(BaseModel):
    """Quality guardrails configuration."""
    diversity: DiversityConfig
    bias_detection: BiasDetectionConfig
    realism: RealismConfig
    rejection_thresholds: RejectionThresholds


class GenerationConfig(BaseModel):
    """Generation settings."""
    num_samples: int = 400
    min_samples: int = 300
    max_samples: int = 500
    batch_size: int = 10
    max_retries: int = 5  # Maximum retries per review before moving to next


class DomainConfig(BaseModel):
    """Domain configuration."""
    name: str
    categories: List[str]
    common_terms: List[str]


class OutputConfig(BaseModel):
    """Output configuration."""
    format: str = "json"
    include_metadata: bool = True
    include_quality_scores: bool = True
    output_dir: str = "output"
    quality_report_path: str = "output/quality_report.md"


class Config(BaseModel):
    """Main configuration model."""
    generation: GenerationConfig
    models: List[ModelConfig]
    personas: List[PersonaConfig]
    review_characteristics: ReviewCharacteristics
    quality_guardrails: QualityGuardrails
    domain: DomainConfig
    output: OutputConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    
    return Config(**config_data)


def save_config(config: Config, config_path: str = "config.yaml") -> None:
    """Save configuration to YAML file."""
    config_file = Path(config_path)
    
    with open(config_file, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
