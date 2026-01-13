"""DSPy programs for structured review generation and validation."""

import os
import dspy
from typing import Dict, Any, Optional
from langsmith import traceable


class ReviewGeneration(dspy.Signature):
    """Generate a realistic SaaS tool review based on persona and requirements."""
    
    persona_description: str = dspy.InputField(desc="Description of the reviewer persona")
    rating: int = dspy.InputField(desc="Target rating (1-5)")
    tool_category: str = dspy.InputField(desc="Category of the SaaS tool")
    requirements: str = dspy.InputField(desc="Specific requirements for the review")
    
    review_text: str = dspy.OutputField(desc="Generated review text (50-800 words)")
    pros: str = dspy.OutputField(desc="Key pros/strengths mentioned")
    cons: str = dspy.OutputField(desc="Key cons/weaknesses mentioned")
    use_case: str = dspy.OutputField(desc="Primary use case described")


class ReviewValidation(dspy.Signature):
    """Validate if a review meets quality and realism criteria."""
    
    review_text: str = dspy.InputField(desc="Review text to validate")
    domain_terms: str = dspy.InputField(desc="Expected domain-specific terms")
    
    is_realistic: bool = dspy.OutputField(desc="Whether the review is realistic")
    realism_score: float = dspy.OutputField(desc="Realism score (0-1)")
    issues: str = dspy.OutputField(desc="List of issues found, if any")
    domain_term_usage: str = dspy.OutputField(desc="Domain terms used in the review")


class SentimentAnalysis(dspy.Signature):
    """Analyze sentiment and rating consistency of a review."""
    
    review_text: str = dspy.InputField(desc="Review text to analyze")
    rating: int = dspy.InputField(desc="Assigned rating")
    
    sentiment_score: float = dspy.OutputField(desc="Sentiment score (-1 to 1, negative to positive)")
    rating_consistency: bool = dspy.OutputField(desc="Whether sentiment matches rating")
    bias_indicators: str = dspy.OutputField(desc="Potential bias indicators found")


class ReviewGenerator:
    """DSPy-based review generator."""
    
    def __init__(self, lm: Optional[dspy.LM] = None):
        """Initialize with a language model."""
        if lm is None:
            # Default to OpenAI if available
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.lm = dspy.OpenAI(model="gpt-4-turbo-preview", api_key=api_key)
            else:
                raise ValueError("No language model configured")
        else:
            self.lm = lm
        
        dspy.configure(lm=self.lm)
        
        # Initialize programs
        self.generate_review = dspy.ChainOfThought(ReviewGeneration)
        self.validate_review = dspy.ChainOfThought(ReviewValidation)
        self.analyze_sentiment = dspy.ChainOfThought(SentimentAnalysis)
    
    @traceable(name="dspy_generate_review")
    def generate(
        self,
        persona: Dict[str, Any],
        rating: int,
        tool_category: str,
        requirements: str
    ) -> Dict[str, Any]:
        """Generate a review using DSPy."""
        persona_desc = self._format_persona(persona)
        
        result = self.generate_review(
            persona_description=persona_desc,
            rating=rating,
            tool_category=tool_category,
            requirements=requirements
        )
        
        return {
            "review_text": result.review_text,
            "pros": result.pros,
            "cons": result.cons,
            "use_case": result.use_case
        }
    
    @traceable(name="dspy_validate_review")
    def validate(
        self,
        review_text: str,
        domain_terms: str
    ) -> Dict[str, Any]:
        """Validate a review using DSPy."""
        result = self.validate_review(
            review_text=review_text,
            domain_terms=domain_terms
        )
        
        return {
            "is_realistic": result.is_realistic,
            "realism_score": float(result.realism_score),
            "issues": result.issues,
            "domain_term_usage": result.domain_term_usage
        }
    
    @traceable(name="dspy_analyze_sentiment")
    def analyze(
        self,
        review_text: str,
        rating: int
    ) -> Dict[str, Any]:
        """Analyze sentiment using DSPy."""
        result = self.analyze_sentiment(
            review_text=review_text,
            rating=rating
        )
        
        return {
            "sentiment_score": float(result.sentiment_score),
            "rating_consistency": result.rating_consistency,
            "bias_indicators": result.bias_indicators
        }
    
    def _format_persona(self, persona: Dict[str, Any]) -> str:
        """Format persona dict into description string."""
        return (
            f"Role: {persona['role']}, "
            f"Company Size: {persona['company_size']}, "
            f"Experience: {persona['experience_years']} years, "
            f"Focus Areas: {', '.join(persona['focus_areas'])}, "
            f"Language Style: {persona['language_style']}"
        )


import os
