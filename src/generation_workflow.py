"""LangGraph workflow for review generation."""

import uuid
import random
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
import time

from langgraph.graph import StateGraph, END
from langsmith import traceable

from src.monitoring import setup_langsmith

from src.config import Config
from src.models import Review, QualityMetrics, GenerationStats
from src.model_providers import ModelProvider, create_provider
from src.quality_guardrails import QualityGuardrail
from src.dspy_programs import ReviewGenerator


class GenerationState(TypedDict):
    """State for the generation workflow."""
    config: Config
    accepted_reviews: List[Review]
    rejected_reviews: List[Review]
    current_review: Optional[Review]
    current_metrics: Optional[QualityMetrics]
    current_persona: Optional[Any]
    current_rating: Optional[int]
    current_model: Optional[str]
    rejection_reason: Optional[str]
    decision: Optional[str]
    stats: GenerationStats
    iteration: int
    max_iterations: int


class ReviewGenerationWorkflow:
    """LangGraph workflow for generating reviews with quality guardrails."""
    
    def __init__(self, config: Config):
        self.config = config
        self.guardrail = QualityGuardrail(config)
        
        # Setup LangSmith monitoring
        self.langsmith_client = setup_langsmith()
        
        # Initialize model providers
        self.providers: Dict[str, ModelProvider] = {}
        for model_config in config.models:
            provider = create_provider(model_config.model_dump())
            self.providers[model_config.name] = provider
        
        # Initialize DSPy generator (optional, can use direct model calls)
        try:
            self.dspy_generator = ReviewGenerator()
        except:
            self.dspy_generator = None
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow - generates ONE review per invocation."""
        workflow = StateGraph(GenerationState)
        
        # Add nodes
        workflow.add_node("select_persona", self._select_persona)
        workflow.add_node("select_model", self._select_model)
        workflow.add_node("generate_review", self._generate_review)
        workflow.add_node("evaluate_quality", self._evaluate_quality)
        workflow.add_node("decide", self._decide)
        workflow.add_node("update_stats", self._update_stats)
        
        # Add edges - linear flow, no loops
        workflow.set_entry_point("select_persona")
        workflow.add_edge("select_persona", "select_model")
        workflow.add_edge("select_model", "generate_review")
        workflow.add_edge("generate_review", "evaluate_quality")
        workflow.add_edge("evaluate_quality", "decide")
        workflow.add_edge("decide", "update_stats")
        workflow.add_edge("update_stats", END)  # Always end after one review
        
        return workflow.compile()
    
    @traceable(name="select_persona")
    def _select_persona(self, state: GenerationState) -> GenerationState:
        """Select a persona for review generation."""
        personas = self.config.personas
        persona = random.choice(personas)
        
        # Select rating based on persona's distribution
        rating = self._sample_rating(persona.rating_distribution)
        
        state["current_persona"] = persona
        state["current_rating"] = rating
        
        return state
    
    @traceable(name="select_model")
    def _select_model(self, state: GenerationState) -> GenerationState:
        """Select a model provider based on weights."""
        models = self.config.models
        weights = [m.weight for m in models]
        selected_model = random.choices(models, weights=weights)[0]
        
        state["current_model"] = selected_model.name
        
        return state
    
    @traceable(name="generate_review")
    def _generate_review(self, state: GenerationState) -> GenerationState:
        """Generate a review using the selected model."""
        persona = state["current_persona"]
        rating = state["current_rating"]
        model_name = state["current_model"]
        
        provider = self.providers[model_name]
        
        # Build prompt
        prompt = self._build_prompt(persona, rating)
        
        # Generate review
        start_time = time.time()
        review_text = provider.generate(prompt)
        generation_time = time.time() - start_time
        
        # Create review object
        review = Review(
            id=str(uuid.uuid4()),
            text=review_text,
            rating=rating,
            persona=persona.name,
            model_used=model_name,
            tool_name=self._select_tool_name(),
            category=random.choice(self.config.domain.categories),
            metadata={
                "generation_time": generation_time,
                "prompt": prompt
            }
        )
        
        state["current_review"] = review
        
        return state
    
    @traceable(name="evaluate_quality")
    def _evaluate_quality(self, state: GenerationState) -> GenerationState:
        """Evaluate quality metrics for the current review."""
        review = state["current_review"]
        accepted_reviews = state.get("accepted_reviews", [])
        
        metrics = self.guardrail.evaluate(review, accepted_reviews)
        
        # Update review with metrics
        review.quality_scores = {
            "diversity": metrics.diversity_score,
            "bias": metrics.bias_score,
            "realism": metrics.realism_score,
            "overall": metrics.overall_quality
        }
        
        state["current_metrics"] = metrics
        state["current_review"] = review
        
        return state
    
    @traceable(name="decide")
    def _decide(self, state: GenerationState) -> GenerationState:
        """Decide whether to accept, reject, or retry."""
        metrics = state["current_metrics"]
        review = state["current_review"]
        stats = state["stats"]
        
        should_reject, reason = self.guardrail.should_reject(metrics)
        
        if should_reject:
            state["rejection_reason"] = reason
            state["decision"] = "reject"
        else:
            state["decision"] = "accept"
        
        return state
    
    # Removed _should_continue and _should_continue_after_update
    # The workflow now generates one review per invocation
    # The outer while loop in generate() handles iteration
    
    @traceable(name="update_stats")
    def _update_stats(self, state: GenerationState) -> GenerationState:
        """Update generation statistics."""
        stats = state["stats"]
        decision = state.get("decision")
        review = state["current_review"]
        model_name = state["current_model"]
        
        if review is None:
            return state
        
        stats.total_generated += 1
        
        if decision == "accept":
            stats.total_accepted += 1
            # Ensure accepted_reviews list exists
            if "accepted_reviews" not in state:
                state["accepted_reviews"] = []
            # Only add if not already present
            if review.id not in [r.id for r in state["accepted_reviews"]]:
                state["accepted_reviews"].append(review)
        else:
            stats.total_rejected += 1
            reason = state.get("rejection_reason", "Unknown")
            if reason not in stats.rejection_reasons:
                stats.rejection_reasons[reason] = 0
            stats.rejection_reasons[reason] += 1
            # Ensure rejected_reviews list exists
            if "rejected_reviews" not in state:
                state["rejected_reviews"] = []
            # Only add if not already present
            if review.id not in [r.id for r in state["rejected_reviews"]]:
                state["rejected_reviews"].append(review)
        
        # Update model stats
        if model_name not in stats.model_stats:
            stats.model_stats[model_name] = {
                "generated": 0,
                "accepted": 0,
                "rejected": 0,
                "total_time": 0.0
            }
        
        model_stat = stats.model_stats[model_name]
        model_stat["generated"] += 1
        if decision == "accept":
            model_stat["accepted"] += 1
        else:
            model_stat["rejected"] += 1
        
        if review.metadata and "generation_time" in review.metadata:
            model_stat["total_time"] += review.metadata["generation_time"]
            stats.total_time_seconds += review.metadata["generation_time"]
        
        state["iteration"] = state.get("iteration", 0) + 1
        
        return state
    
    def _build_prompt(self, persona: Any, rating: int) -> str:
        """Build generation prompt."""
        char_config = self.config.review_characteristics
        
        prompt = f"""Write an authentic SaaS tool review from the perspective of:
- Role: {persona.role}
- Company Size: {persona.company_size}
- Experience: {persona.experience_years} years
- Focus Areas: {', '.join(persona.focus_areas)}
- Language Style: {persona.language_style}

Requirements:
- Rating: {rating}/5 stars
- Length: {char_config.min_length}-{char_config.max_length} characters
- Include specific pros and cons
- Mention use case and context
- Use domain-appropriate terminology
- Write in {persona.language_style} style
- Be authentic and realistic

Write the review:"""
        
        return prompt
    
    def _sample_rating(self, distribution: Dict[int, float]) -> int:
        """Sample a rating from distribution."""
        ratings = list(distribution.keys())
        probabilities = [distribution[r] for r in ratings]
        return random.choices(ratings, weights=probabilities)[0]
    
    def _select_tool_name(self) -> str:
        """Select a tool name (placeholder - can be enhanced)."""
        tool_names = [
            "ProjectFlow", "TeamSync", "DataViz Pro", "CloudConnect",
            "TaskMaster", "AnalyticsHub", "CommBridge", "WorkSpace Pro"
        ]
        return random.choice(tool_names)
    
    @traceable(name="run_generation")
    def generate(self, num_samples: Optional[int] = None) -> List[Review]:
        """Run the generation workflow."""
        if num_samples is None:
            num_samples = self.config.generation.num_samples
        
        # Initialize state
        stats = GenerationStats()
        accepted_reviews = []
        rejected_reviews = []
        review_attempt = 0  # Track how many reviews we've attempted (not retries)
        max_retries_per_review = 5  # Maximum retries per review (user requirement)
        max_total_reviews = num_samples * 10  # Safety limit to prevent infinite loops
        
        print(f"Target: {num_samples} accepted reviews (max {max_retries_per_review} retries per review)")
        
        # Run generation loop - try to get one accepted review at a time
        while len(accepted_reviews) < num_samples and review_attempt < max_total_reviews:
            review_attempt += 1
            retry_count = 0
            review_accepted = False
            
            # Try to generate one accepted review (with up to max_retries_per_review attempts)
            while retry_count < max_retries_per_review and not review_accepted:
                retry_count += 1
                
                # Create state for this attempt
                state: GenerationState = {
                    "config": self.config,
                    "accepted_reviews": accepted_reviews.copy(),
                    "rejected_reviews": rejected_reviews.copy(),
                    "current_review": None,
                    "current_metrics": None,
                    "current_persona": None,
                    "current_rating": None,
                    "current_model": None,
                    "rejection_reason": None,
                    "decision": None,
                    "stats": stats,
                    "iteration": review_attempt,
                    "max_iterations": max_total_reviews
                }
                
                # Run workflow for one review (linear flow, no recursion)
                try:
                    # Set recursion limit to 10 (should be enough for linear flow)
                    final_state = self.workflow.invoke(state, config={"recursion_limit": 10})
                    
                    # Get the current review from final state
                    current_review = final_state.get("current_review")
                    decision = final_state.get("decision")
                    
                    if current_review:
                        # Add review to appropriate list based on decision
                        if decision == "accept":
                            # Check if not already in accepted list
                            if current_review.id not in [r.id for r in accepted_reviews]:
                                accepted_reviews.append(current_review)
                                review_accepted = True
                                if retry_count > 1:
                                    print(f"  ✓ Accepted review {len(accepted_reviews)}/{num_samples} (after {retry_count} attempts)")
                                else:
                                    print(f"  ✓ Accepted review {len(accepted_reviews)}/{num_samples}")
                        else:
                            # Rejected
                            if current_review.id not in [r.id for r in rejected_reviews]:
                                rejected_reviews.append(current_review)
                                reason = final_state.get("rejection_reason", "Quality check failed")
                                if retry_count < max_retries_per_review:
                                    print(f"  ✗ Rejected (attempt {retry_count}/{max_retries_per_review}): {reason[:50]}... retrying")
                                else:
                                    print(f"  ✗ Rejected (attempt {retry_count}/{max_retries_per_review}): {reason[:50]}... moving to next review")
                    
                    # Update stats from final state
                    stats = final_state.get("stats", stats)
                    
                except Exception as e:
                    print(f"  ⚠ Error in attempt {retry_count}: {e}")
                    if retry_count >= max_retries_per_review:
                        print(f"  → Max retries reached for this review, moving to next")
                    continue
                
                # Break if we have enough accepted reviews
                if len(accepted_reviews) >= num_samples:
                    break
            
            # Progress indicator
            if review_attempt % 5 == 0 or len(accepted_reviews) >= num_samples:
                print(f"Progress: {len(accepted_reviews)}/{num_samples} accepted, {len(rejected_reviews)} rejected, {review_attempt} reviews attempted")
            
            # Break early if we have enough accepted reviews
            if len(accepted_reviews) >= num_samples:
                break
        
        # Calculate final stats
        if stats.total_generated > 0:
            stats.avg_time_per_review = stats.total_time_seconds / stats.total_generated
        
        print(f"\n✓ Generation complete: {len(accepted_reviews)} accepted, {len(rejected_reviews)} rejected, {review_attempt} reviews attempted")
        
        if len(accepted_reviews) < num_samples:
            print(f"⚠ Warning: Only generated {len(accepted_reviews)}/{num_samples} accepted reviews. Consider:")
            print(f"  - Adjusting quality thresholds in config.yaml")
            print(f"  - Increasing max_retries_per_review (currently {max_retries_per_review})")
        
        return accepted_reviews
