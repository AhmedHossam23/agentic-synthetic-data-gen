"""Quality reporting and metrics analysis."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from src.models import Review, GenerationStats, ComparisonMetrics, QualityMetrics
from src.config import Config

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class QualityReporter:
    """Generate quality reports and metrics."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_report(
        self,
        reviews: List[Review],
        stats: GenerationStats,
        real_reviews: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate a comprehensive quality report in Markdown."""
        report_lines = []
        
        # Header
        report_lines.append("# Synthetic Data Generation Quality Report")
        report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Domain:** {self.config.domain.name}")
        report_lines.append(f"**Total Samples:** {len(reviews)}")
        report_lines.append("\n---\n")
        
        # Generation Statistics
        report_lines.append("## Generation Statistics\n")
        report_lines.append(f"- **Total Generated:** {stats.total_generated}")
        report_lines.append(f"- **Accepted:** {stats.total_accepted}")
        report_lines.append(f"- **Rejected:** {stats.total_rejected}")
        report_lines.append(f"- **Acceptance Rate:** {(stats.total_accepted / stats.total_generated * 100):.2f}%")
        report_lines.append(f"- **Total Time:** {stats.total_time_seconds:.2f} seconds")
        report_lines.append(f"- **Avg Time per Review:** {stats.avg_time_per_review:.2f} seconds")
        report_lines.append("\n")
        
        # Rejection Reasons
        if stats.rejection_reasons:
            report_lines.append("### Rejection Reasons\n")
            for reason, count in stats.rejection_reasons.items():
                report_lines.append(f"- {reason}: {count}")
            report_lines.append("\n")
        
        # Model Performance
        if stats.model_stats:
            report_lines.append("## Model Performance\n")
            for model_name, model_stat in stats.model_stats.items():
                report_lines.append(f"### {model_name}\n")
                report_lines.append(f"- Generated: {model_stat['generated']}")
                report_lines.append(f"- Accepted: {model_stat['accepted']}")
                report_lines.append(f"- Rejected: {model_stat['rejected']}")
                if model_stat['generated'] > 0:
                    acceptance_rate = model_stat['accepted'] / model_stat['generated'] * 100
                    report_lines.append(f"- Acceptance Rate: {acceptance_rate:.2f}%")
                if model_stat['accepted'] > 0:
                    avg_time = model_stat['total_time'] / model_stat['accepted']
                    report_lines.append(f"- Avg Time per Accepted Review: {avg_time:.2f} seconds")
                report_lines.append("\n")
        
        # Quality Metrics Summary
        report_lines.append("## Quality Metrics Summary\n")
        quality_scores = [r.quality_scores for r in reviews if r.quality_scores]
        
        if quality_scores:
            avg_diversity = np.mean([q.get("diversity", 0) for q in quality_scores])
            avg_bias = np.mean([q.get("bias", 0) for q in quality_scores])
            avg_realism = np.mean([q.get("realism", 0) for q in quality_scores])
            avg_overall = np.mean([q.get("overall", 0) for q in quality_scores])
            
            report_lines.append(f"- **Average Diversity Score:** {avg_diversity:.3f}")
            report_lines.append(f"- **Average Bias Score:** {avg_bias:.3f}")
            report_lines.append(f"- **Average Realism Score:** {avg_realism:.3f}")
            report_lines.append(f"- **Average Overall Quality:** {avg_overall:.3f}")
            report_lines.append("\n")
        
        # Rating Distribution
        report_lines.append("## Rating Distribution\n")
        ratings = [r.rating for r in reviews]
        if HAS_PANDAS:
            rating_counts = pd.Series(ratings).value_counts().sort_index()
            for rating, count in rating_counts.items():
                percentage = count / len(ratings) * 100
                report_lines.append(f"- **{rating} stars:** {count} ({percentage:.1f}%)")
        else:
            # Fallback without pandas
            from collections import Counter
            rating_counts = Counter(ratings)
            for rating in sorted(rating_counts.keys()):
                count = rating_counts[rating]
                percentage = count / len(ratings) * 100
                report_lines.append(f"- **{rating} stars:** {count} ({percentage:.1f}%)")
        report_lines.append("\n")
        
        # Persona Distribution
        report_lines.append("## Persona Distribution\n")
        personas = [r.persona for r in reviews]
        if HAS_PANDAS:
            persona_counts = pd.Series(personas).value_counts()
            for persona, count in persona_counts.items():
                percentage = count / len(personas) * 100
                report_lines.append(f"- **{persona}:** {count} ({percentage:.1f}%)")
        else:
            # Fallback without pandas
            from collections import Counter
            persona_counts = Counter(personas)
            for persona, count in persona_counts.items():
                percentage = count / len(personas) * 100
                report_lines.append(f"- **{persona}:** {count} ({percentage:.1f}%)")
        report_lines.append("\n")
        
        # Comparison with Real Reviews
        if real_reviews:
            comparison = self._compare_with_real(reviews, real_reviews)
            report_lines.append("## Comparison with Real Reviews\n")
            report_lines.append(f"**Real Reviews Analyzed:** {len(real_reviews)}\n")
            
            # Similarity metrics
            if comparison.similarity_metrics:
                report_lines.append("### Similarity Metrics\n")
                for metric, value in comparison.similarity_metrics.items():
                    report_lines.append(f"- **{metric}:** {value:.3f}")
                report_lines.append("\n")
            
            # Differences
            if comparison.differences:
                report_lines.append("### Key Differences\n")
                for diff in comparison.differences:
                    report_lines.append(f"- {diff}")
                report_lines.append("\n")
        
        # Recommendations
        report_lines.append("## Recommendations\n")
        if quality_scores:
            if avg_diversity < 0.5:
                report_lines.append("- ⚠️ Diversity scores are low. Consider increasing variety in prompts and personas.")
            if avg_bias > 0.5:
                report_lines.append("- ⚠️ Bias scores are high. Review rating distributions and sentiment patterns.")
            if avg_realism < 0.6:
                report_lines.append("- ⚠️ Realism scores could be improved. Check domain term usage and readability.")
        
        if stats.total_rejected > stats.total_accepted * 0.3:
            report_lines.append("- ⚠️ High rejection rate. Consider adjusting quality thresholds or improving prompts.")
        
        report_lines.append("\n---\n")
        report_lines.append("*Report generated by Synthetic Data Generator*")
        
        return "\n".join(report_lines)
    
    def _compare_with_real(
        self,
        synthetic_reviews: List[Review],
        real_reviews: List[Dict[str, Any]]
    ) -> ComparisonMetrics:
        """Compare synthetic reviews with real reviews."""
        # Rating distribution comparison
        synth_ratings = [r.rating for r in synthetic_reviews]
        real_ratings = [r.get("rating", 0) for r in real_reviews if "rating" in r]
        
        if HAS_PANDAS:
            synth_rating_dist = pd.Series(synth_ratings).value_counts(normalize=True).sort_index()
            real_rating_dist = pd.Series(real_ratings).value_counts(normalize=True).sort_index()
        else:
            from collections import Counter
            synth_counter = Counter(synth_ratings)
            real_counter = Counter(real_ratings)
            total_synth = sum(synth_counter.values())
            total_real = sum(real_counter.values())
            synth_rating_dist = {k: v/total_synth for k, v in synth_counter.items()}
            real_rating_dist = {k: v/total_real for k, v in real_counter.items()}
        
        # Calculate similarity
        if HAS_PANDAS:
            all_ratings = set(synth_rating_dist.index) | set(real_rating_dist.index)
            similarity = sum(
                min(synth_rating_dist.get(r, 0), real_rating_dist.get(r, 0))
                for r in all_ratings
            )
        else:
            all_ratings = set(synth_rating_dist.keys()) | set(real_rating_dist.keys())
            similarity = sum(
                min(synth_rating_dist.get(r, 0), real_rating_dist.get(r, 0))
                for r in all_ratings
            )
        
        # Length comparison
        synth_lengths = [len(r.text) for r in synthetic_reviews]
        real_lengths = [len(r.get("text", "")) for r in real_reviews if "text" in r]
        
        avg_synth_length = np.mean(synth_lengths) if synth_lengths else 0
        avg_real_length = np.mean(real_lengths) if real_lengths else 0
        
        # Differences
        differences = []
        if abs(avg_synth_length - avg_real_length) > 100:
            differences.append(f"Average length differs: synthetic={avg_synth_length:.0f}, real={avg_real_length:.0f}")
        
        if similarity < 0.7:
            differences.append(f"Rating distribution differs significantly (similarity={similarity:.2f})")
        
        return ComparisonMetrics(
            synthetic_stats={
                "avg_length": avg_synth_length,
                "rating_distribution": synth_rating_dist if isinstance(synth_rating_dist, dict) else synth_rating_dist.to_dict()
            },
            real_stats={
                "avg_length": avg_real_length,
                "rating_distribution": real_rating_dist if isinstance(real_rating_dist, dict) else real_rating_dist.to_dict()
            },
            similarity_metrics={
                "rating_distribution_similarity": similarity,
                "length_similarity": 1.0 - abs(avg_synth_length - avg_real_length) / max(avg_synth_length, avg_real_length, 1)
            },
            differences=differences
        )
    
    def save_report(self, report: str, output_path: str) -> None:
        """Save report to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            f.write(report)
    
    def save_reviews_json(self, reviews: List[Review], output_path: str) -> None:
        """Save reviews to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        reviews_data = [r.model_dump() for r in reviews]
        
        with open(output_file, "w") as f:
            json.dump(reviews_data, f, indent=2, default=str)


from typing import Optional
