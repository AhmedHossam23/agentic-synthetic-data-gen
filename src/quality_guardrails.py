"""Quality guardrails for review generation."""

import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import warnings

from src.config import QualityGuardrails, Config
from src.models import Review, QualityMetrics

# Optional imports with fallbacks
try:
    from langsmith import traceable
    HAS_LANGSMITH = True
except ImportError:
    HAS_LANGSMITH = False
    # Fallback decorator that does nothing
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    warnings.warn("textstat not available. Readability scores will use fallback method.")

# Gemini embeddings
_gemini_embedding_model = None

def _get_gemini_embedding_model():
    """Lazy load Gemini embedding model."""
    global _gemini_embedding_model
    if _gemini_embedding_model is None:
        try:
            import os
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                warnings.warn("GOOGLE_API_KEY not found. Semantic similarity will use fallback method.")
                _gemini_embedding_model = False
                return None
            
            genai.configure(api_key=api_key)
            # Use Gemini embedding model
            _gemini_embedding_model = genai
        except ImportError as e:
            warnings.warn(f"Could not import google.generativeai: {e}. Semantic similarity will use fallback method.")
            _gemini_embedding_model = False
        except Exception as e:
            warnings.warn(f"Could not initialize Gemini embeddings: {e}. Semantic similarity will use fallback method.")
            _gemini_embedding_model = False
    return _gemini_embedding_model if _gemini_embedding_model is not False else None


class QualityGuardrail:
    """Quality guardrail system for review validation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.guardrails = config.quality_guardrails
        
        # Initialize Gemini embeddings for semantic similarity (lazy)
        self.gemini_model = _get_gemini_embedding_model()
        self.use_semantic_similarity = self.gemini_model is not None
        
        # Domain terms for realism checking
        self.domain_terms = set(term.lower() for term in config.domain.common_terms)
        
        # Cache for embeddings (review_id -> embedding)
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Limit comparisons for performance (compare to last N reviews only)
        self.max_comparison_reviews = 20
    
    @traceable(name="calculate_quality_metrics")
    def evaluate(self, review: Review, existing_reviews: List[Review]) -> QualityMetrics:
        """Evaluate quality metrics for a review."""
        diversity_score = self._calculate_diversity(review, existing_reviews)
        bias_score = self._calculate_bias(review, existing_reviews)
        realism_score = self._calculate_realism(review)
        
        # Overall quality is weighted average
        # Note: bias_score is inverted (lower is better), so we use (1 - bias_score)
        overall_quality = (
            diversity_score * 0.35 +
            (1.0 - bias_score) * 0.35 +  # Invert bias (lower bias = higher quality)
            realism_score * 0.3
        )
        
        return QualityMetrics(
            diversity_score=diversity_score,
            bias_score=bias_score,
            realism_score=realism_score,
            overall_quality=overall_quality,
            vocabulary_overlap=self._vocabulary_overlap(review, existing_reviews),
            semantic_similarity=self._semantic_similarity(review, existing_reviews),
            sentiment_score=self._sentiment_score(review),
            readability_score=self._readability_score(review),
            domain_term_ratio=self._domain_term_ratio(review)
        )
    
    def should_reject(self, metrics: QualityMetrics) -> Tuple[bool, str]:
        """Determine if a review should be rejected."""
        thresholds = self.guardrails.rejection_thresholds
        
        reasons = []
        
        if metrics.diversity_score < thresholds.diversity_score:
            reasons.append(f"Low diversity score: {metrics.diversity_score:.2f}")
        
        if metrics.bias_score > thresholds.bias_score:
            reasons.append(f"High bias score: {metrics.bias_score:.2f}")
        
        if metrics.realism_score < thresholds.realism_score:
            reasons.append(f"Low realism score: {metrics.realism_score:.2f}")
        
        if metrics.overall_quality < thresholds.overall_quality:
            reasons.append(f"Low overall quality: {metrics.overall_quality:.2f}")
        
        should_reject = len(reasons) > 0
        
        return should_reject, "; ".join(reasons) if reasons else "Accepted"
    
    def _calculate_diversity(self, review: Review, existing_reviews: List[Review]) -> float:
        """Calculate diversity score (higher is better)."""
        if not existing_reviews:
            return 1.0  # First review is always diverse
        
        # With very few reviews, be more lenient
        if len(existing_reviews) < 3:
            # For first few reviews, only check if extremely similar
            vocab_overlap = self._vocabulary_overlap(review, existing_reviews)
            semantic_sim = self._semantic_similarity(review, existing_reviews)
            
            # If overlap/similarity is very high (>0.8), penalize
            if vocab_overlap > 0.8 or semantic_sim > 0.8:
                return 0.3  # Low diversity if too similar
            else:
                return 0.7  # Good enough diversity for early reviews
        
        # Vocabulary overlap
        vocab_overlap = self._vocabulary_overlap(review, existing_reviews)
        
        # Semantic similarity
        semantic_sim = self._semantic_similarity(review, existing_reviews)
        
        # Diversity is HIGHER when overlap/similarity is LOWER
        # Normalize: low overlap = high diversity, high overlap = low diversity
        vocab_diversity = 1.0 - min(1.0, vocab_overlap * 1.5)  # Scale overlap to diversity
        semantic_diversity = 1.0 - min(1.0, semantic_sim * 1.2)  # Scale similarity to diversity
        
        # Check if within acceptable ranges
        div_config = self.guardrails.diversity
        if (div_config.min_vocabulary_overlap <= vocab_overlap <= div_config.max_vocabulary_overlap and
            div_config.min_semantic_similarity <= semantic_sim <= div_config.max_semantic_similarity):
            # Within acceptable range - good diversity
            diversity = (vocab_diversity + semantic_diversity) / 2
        else:
            # Outside acceptable range - check if too similar or too different
            if vocab_overlap > div_config.max_vocabulary_overlap or semantic_sim > div_config.max_semantic_similarity:
                # Too similar - low diversity
                diversity = min(vocab_diversity, semantic_diversity) * 0.6
            else:
                # Too different - actually good for diversity, but might be unrealistic
                diversity = (vocab_diversity + semantic_diversity) / 2 * 0.8
        
        return max(0.0, min(1.0, diversity))
    
    def _calculate_bias(self, review: Review, existing_reviews: List[Review]) -> float:
        """Calculate bias score (lower is better)."""
        if not existing_reviews:
            return 0.0  # First review has no bias
        
        # With very few reviews, be more lenient
        if len(existing_reviews) < 3:
            # For first few reviews, only flag extreme bias
            sentiment = self._sentiment_score(review)
            existing_sentiments = [self._sentiment_score(r) for r in existing_reviews]
            
            if existing_sentiments:
                avg_sentiment = np.mean(existing_sentiments)
                sentiment_skew = abs(sentiment - avg_sentiment)
                
                # Only flag if extremely skewed (>0.8 difference)
                if sentiment_skew > 0.8:
                    return 0.6
                else:
                    return 0.2  # Low bias for early reviews
        
        # Sentiment skew
        sentiment = self._sentiment_score(review)
        existing_sentiments = [self._sentiment_score(r) for r in existing_reviews]
        
        bias_scores = []
        
        if existing_sentiments:
            avg_sentiment = np.mean(existing_sentiments)
            sentiment_skew = abs(sentiment - avg_sentiment)
            
            # Check if exceeds threshold
            max_skew = self.guardrails.bias_detection.max_sentiment_skew
            if sentiment_skew > max_skew:
                bias_scores.append(min(1.0, sentiment_skew / max_skew))
            else:
                # Normalize: small skew is good (low bias)
                bias_scores.append(sentiment_skew / max_skew * 0.3)
        
        # Rating variance check - but be more lenient
        ratings = [r.rating for r in existing_reviews] + [review.rating]
        rating_variance = np.var(ratings)
        min_variance = self.guardrails.bias_detection.min_rating_variance
        
        # Only flag if variance is extremely low AND we have enough reviews
        if len(ratings) >= 5 and rating_variance < min_variance * 0.5:
            bias_scores.append(0.6)  # Moderate bias if very low variance with many reviews
        elif rating_variance < min_variance:
            bias_scores.append(0.3)  # Low bias if slightly below threshold
        
        # Return maximum bias score, or average if multiple factors
        if bias_scores:
            return min(1.0, max(bias_scores))
        else:
            return 0.2  # Default low bias
    
    def _readability_score(self, review: Review) -> float:
        """Calculate readability score using textstat or fallback."""
        if HAS_TEXTSTAT:
            return textstat.flesch_reading_ease(review.text)
        else:
            # Fallback: Simple heuristic based on sentence length and word length
            sentences = review.text.split('.')
            if not sentences:
                return 50.0  # Default middle score
            
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            avg_word_length = sum(len(word) for word in review.text.split()) / max(1, len(review.text.split()))
            
            # Flesch-like heuristic: shorter sentences and words = higher readability
            # Normalize to 0-100 scale
            score = 100 - (avg_sentence_length * 1.5) - (avg_word_length * 10)
            return max(0.0, min(100.0, score))
    
    def _calculate_realism(self, review: Review) -> float:
        """Calculate realism score."""
        realism_config = self.guardrails.realism
        
        # Readability check
        readability = self._readability_score(review)
        if (realism_config.min_readability_score <= readability <= realism_config.max_readability_score):
            readability_score = 1.0
        else:
            # Penalize if outside range
            if readability < realism_config.min_readability_score:
                readability_score = readability / realism_config.min_readability_score
            else:
                readability_score = (100 - readability) / (100 - realism_config.max_readability_score)
            readability_score = max(0.0, min(1.0, readability_score))
        
        # Domain term usage
        domain_term_ratio = self._domain_term_ratio(review)
        min_ratio = realism_config.min_domain_term_ratio
        
        if domain_term_ratio >= min_ratio:
            domain_score = 1.0
        else:
            domain_score = domain_term_ratio / min_ratio
        
        # Length check
        text_length = len(review.text.split())
        char_config = self.config.review_characteristics
        if char_config.min_length <= len(review.text) <= char_config.max_length:
            length_score = 1.0
        else:
            length_score = 0.5
        
        # Combined realism score
        realism = (readability_score * 0.4 + domain_score * 0.4 + length_score * 0.2)
        
        return max(0.0, min(1.0, realism))
    
    def _vocabulary_overlap(self, review: Review, existing_reviews: List[Review]) -> float:
        """Calculate vocabulary overlap with existing reviews."""
        if not existing_reviews:
            return 0.0
        
        # Limit to recent reviews for performance
        reviews_to_compare = existing_reviews[-self.max_comparison_reviews:]
        
        # Extract unique words (lowercase, no punctuation)
        review_words = set(re.findall(r'\b\w+\b', review.text.lower()))
        
        all_existing_words = set()
        for r in reviews_to_compare:
            all_existing_words.update(re.findall(r'\b\w+\b', r.text.lower()))
        
        if not all_existing_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = review_words & all_existing_words
        union = review_words | all_existing_words
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _semantic_similarity(self, review: Review, existing_reviews: List[Review]) -> float:
        """Calculate average semantic similarity with existing reviews using Gemini embeddings."""
        if not existing_reviews:
            return 0.0
        
        # Limit to recent reviews for performance (O(n) instead of O(n²))
        reviews_to_compare = existing_reviews[-self.max_comparison_reviews:]
        
        # Use Gemini embeddings if available
        if self.use_semantic_similarity and self.gemini_model:
            try:
                import google.generativeai as genai
                
                # Get embedding for new review (with caching)
                review_embedding = self._get_gemini_embedding_cached(review.id, review.text)
                if review_embedding is None:
                    return self._fallback_semantic_similarity(review, reviews_to_compare)
                
                # Get embeddings for existing reviews (with caching)
                existing_embeddings = []
                for existing_review in reviews_to_compare:
                    emb = self._get_gemini_embedding_cached(existing_review.id, existing_review.text)
                    if emb is not None:
                        existing_embeddings.append(emb)
                
                if not existing_embeddings:
                    return self._fallback_semantic_similarity(review, reviews_to_compare)
                
                # Calculate cosine similarity
                similarities = []
                for existing_emb in existing_embeddings:
                    similarity = self._cosine_similarity(review_embedding, existing_emb)
                    similarities.append(similarity)
                
                return float(np.mean(similarities)) if similarities else 0.0
            except Exception as e:
                warnings.warn(f"Error in Gemini semantic similarity calculation: {e}. Using fallback.")
        
        # Fallback: Simple word overlap-based similarity
        return self._fallback_semantic_similarity(review, reviews_to_compare)
    
    def _get_gemini_embedding_cached(self, review_id: str, text: str) -> Optional[List[float]]:
        """Get embedding with caching to avoid repeated API calls."""
        if review_id in self._embedding_cache:
            return self._embedding_cache[review_id]
        
        embedding = self._get_gemini_embedding(text)
        if embedding is not None:
            self._embedding_cache[review_id] = embedding
        return embedding
    
    def _get_gemini_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Gemini."""
        try:
            import google.generativeai as genai
            
            # Use the embedding model
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"  # or "retrieval_query", "semantic_similarity", etc.
            )
            return result['embedding']
        except Exception as e:
            warnings.warn(f"Error getting Gemini embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def _fallback_semantic_similarity(self, review: Review, existing_reviews: List[Review]) -> float:
        """Fallback semantic similarity using word overlap."""
        # Limit comparisons for performance
        reviews_to_compare = existing_reviews[-self.max_comparison_reviews:]
        
        review_words = set(re.findall(r'\b\w+\b', review.text.lower()))
        
        similarities = []
        for existing_review in reviews_to_compare:
            existing_words = set(re.findall(r'\b\w+\b', existing_review.text.lower()))
            
            if not review_words or not existing_words:
                similarities.append(0.0)
                continue
            
            # Jaccard similarity
            intersection = review_words & existing_words
            union = review_words | existing_words
            similarity = len(intersection) / len(union) if union else 0.0
            similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _sentiment_score(self, review: Review) -> float:
        """Calculate sentiment score (-1 to 1)."""
        # Simple sentiment based on rating and keywords
        rating_sentiment = (review.rating - 3) / 2.0  # Map 1-5 to -1 to 1
        
        # Check for positive/negative keywords
        positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'fantastic', 'wonderful']
        negative_words = ['terrible', 'awful', 'horrible', 'hate', 'disappointing', 'poor', 'bad']
        
        text_lower = review.text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        keyword_sentiment = (positive_count - negative_count) / max(1, positive_count + negative_count)
        
        # Combine rating and keyword sentiment
        sentiment = (rating_sentiment * 0.7 + keyword_sentiment * 0.3)
        
        return max(-1.0, min(1.0, sentiment))
    
    def _domain_term_ratio(self, review: Review) -> float:
        """Calculate ratio of domain-specific terms in review."""
        words = re.findall(r'\b\w+\b', review.text.lower())
        
        if not words:
            return 0.0
        
        domain_word_count = sum(1 for word in words if word in self.domain_terms)
        
        return domain_word_count / len(words)
