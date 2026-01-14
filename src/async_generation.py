"""Async generation workflow for parallel review generation."""

import uuid
import random
import asyncio
import time
from typing import Dict, Any, List, Optional, Any as AnyType
from datetime import datetime

from src.config import Config
from src.models import Review, QualityMetrics, GenerationStats
from src.async_model_providers import AsyncModelProvider, create_async_provider
from src.quality_guardrails import QualityGuardrail
from src.monitoring import setup_langsmith


class AsyncReviewGenerator:
    """Async review generator with parallel processing and rate limiting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.guardrail = QualityGuardrail(config)
        
        # Setup LangSmith monitoring
        self.langsmith_client = setup_langsmith()
        
        # Initialize async model providers
        self.providers: Dict[str, AsyncModelProvider] = {}
        for model_config in config.models:
            provider = create_async_provider(model_config.model_dump())
            self.providers[model_config.name] = provider
        
        # Rate limiting: semaphore to limit concurrent API calls
        self.max_concurrent = config.generation.batch_size
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Thread-safe locks for shared state
        self.lock = asyncio.Lock()
    
    async def _generate_single_review(
        self,
        persona: Any,
        rating: int,
        model_name: str,
        accepted_reviews: List[Review],
        review_id: str
    ) -> Optional[Review]:
        """Generate a single review asynchronously."""
        async with self.semaphore:  # Rate limiting
            try:
                provider = self.providers[model_name]
                
                # Build prompt
                prompt = self._build_prompt(persona, rating)
                
                # Generate review with timeout (60 seconds per review)
                start_time = time.time()
                try:
                    review_text = await asyncio.wait_for(
                        provider.generate(prompt),
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    print(f"  ⚠ Timeout generating review with {model_name}")
                    return None
                generation_time = time.time() - start_time
                
                # Create review object
                review = Review(
                    id=review_id,
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
                
                # Evaluate quality (thread-safe)
                async with self.lock:
                    # Get current accepted reviews snapshot
                    current_accepted = accepted_reviews.copy()
                
                # Run blocking quality check in thread pool to not block async event loop
                loop = asyncio.get_event_loop()
                metrics = await loop.run_in_executor(
                    None,  # Use default thread pool
                    lambda: self.guardrail.evaluate(review, current_accepted)
                )
                
                # Update review with metrics
                review.quality_scores = {
                    "diversity": metrics.diversity_score,
                    "bias": metrics.bias_score,
                    "realism": metrics.realism_score,
                    "overall": metrics.overall_quality
                }
                
                # Decide acceptance
                should_reject, reason = self.guardrail.should_reject(metrics)
                
                if not should_reject:
                    review.metadata["decision"] = "accept"
                    return review
                else:
                    review.metadata["decision"] = "reject"
                    review.metadata["rejection_reason"] = reason
                    return None  # Rejected
                    
            except Exception as e:
                print(f"  ⚠ Error generating review: {e}")
                return None
    
    async def _generate_review_with_retries(
        self,
        persona: Any,
        rating: int,
        model_name: str,
        accepted_reviews: List[Review],
        max_retries: int = 5
    ) -> Optional[Review]:
        """Generate a review with retries."""
        for attempt in range(1, max_retries + 1):
            review_id = f"{persona.name}_{rating}_{model_name}_{uuid.uuid4().hex[:8]}"
            try:
                review = await asyncio.wait_for(
                    self._generate_single_review(
                        persona, rating, model_name, accepted_reviews, review_id
                    ),
                    timeout=90.0  # 90 seconds per attempt (including retries)
                )
                
                if review:
                    return review
                else:
                    # Small delay between retries to avoid rate limits
                    if attempt < max_retries:
                        await asyncio.sleep(0.2)
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    continue  # Try next attempt
                else:
                    return None  # Failed after max retries
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(0.2)
                    continue
                else:
                    return None
        
        return None  # Failed after max retries
    
    async def generate_batch(
        self,
        batch_size: int,
        accepted_reviews_snapshot: List[Review],
        max_retries: int = 5
    ) -> List[Review]:
        """Generate a batch of reviews in parallel."""
        # Select personas and models for this batch
        personas = self.config.personas
        models = self.config.models
        
        # Create tasks for parallel generation
        tasks = []
        for _ in range(batch_size):
            persona = random.choice(personas)
            rating = self._sample_rating(persona.rating_distribution)
            
            # Select model based on weights
            weights = [m.weight for m in models]
            selected_model = random.choices(models, weights=weights)[0]
            
            # Pass snapshot to avoid race conditions
            task = self._generate_review_with_retries(
                persona, rating, selected_model.name, accepted_reviews_snapshot, max_retries
            )
            tasks.append(task)
        
        # Execute all tasks in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300.0  # 5 minutes for entire batch
            )
        except asyncio.TimeoutError:
            print(f"  ⚠ Batch generation timeout, some tasks may be incomplete")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = []
        
        # Filter out None and exceptions
        accepted = []
        for i, result in enumerate(results):
            if isinstance(result, Review):
                accepted.append(result)
            elif isinstance(result, Exception):
                print(f"  ⚠ Exception in task {i+1}: {type(result).__name__}: {result}")
            elif result is None:
                # Rejected review
                pass
        
        return accepted
    
    async def generate(self, num_samples: Optional[int] = None) -> List[Review]:
        """Generate reviews asynchronously with parallel processing."""
        if num_samples is None:
            num_samples = self.config.generation.num_samples
        
        stats = GenerationStats()
        accepted_reviews: List[Review] = []
        rejected_count = 0
        max_retries = self.config.generation.max_retries
        batch_size = self.config.generation.batch_size
        
        print(f"Target: {num_samples} accepted reviews")
        print(f"Using async generation with batch size: {batch_size}, max retries: {max_retries}")
        
        total_attempts = 0
        max_total_attempts = num_samples * (max_retries + 1) * 2  # Safety limit
        batch_number = 0
        
        # Generate in batches until we have enough
        while len(accepted_reviews) < num_samples and total_attempts < max_total_attempts:
            batch_number += 1
            # Calculate how many we still need
            needed = num_samples - len(accepted_reviews)
            # Generate batch with some extra for expected rejections
            current_batch_size = min(batch_size, max(needed, batch_size))
            
            print(f"\n📦 Starting batch {batch_number}: generating {current_batch_size} reviews "
                  f"(need {needed} more, have {len(accepted_reviews)}/{num_samples})...")
            
            # Get snapshot of accepted reviews for this batch (quick lock)
            async with self.lock:
                current_accepted = accepted_reviews.copy()
            
            # Generate batch (parallel) - use snapshot to avoid race conditions
            batch_start = time.time()
            try:
                # Add timeout for entire batch (5 minutes max)
                batch_results = await asyncio.wait_for(
                    self.generate_batch(
                        current_batch_size,
                        current_accepted.copy(),  # Pass copy of snapshot
                        max_retries
                    ),
                    timeout=300.0  # 5 minutes per batch
                )
            except asyncio.TimeoutError:
                print(f"  ⚠ Batch timeout after 5 minutes, continuing with next batch...")
                batch_results = []
            except Exception as e:
                print(f"  ⚠ Batch error: {e}, continuing with next batch...")
                batch_results = []
            batch_time = time.time() - batch_start
            
            # Update accepted reviews (thread-safe)
            async with self.lock:
                new_accepted_count = 0
                for review in batch_results:
                    if review and review.id not in [r.id for r in accepted_reviews]:
                        accepted_reviews.append(review)
                        new_accepted_count += 1
                        stats.total_accepted += 1
                        print(f"  ✓ Accepted review {len(accepted_reviews)}/{num_samples}")
                    else:
                        rejected_count += 1
                
                # Update stats
                stats.total_generated += len(batch_results)
                stats.total_time_seconds += batch_time
            
            total_attempts += current_batch_size
            
            # Progress update
            print(f"📊 Batch {batch_number} complete: {new_accepted_count} new accepted from {len(batch_results)} results "
                  f"(took {batch_time:.1f}s)")
            if len(accepted_reviews) % 10 == 0 or len(accepted_reviews) >= num_samples:
                print(f"📈 Overall Progress: {len(accepted_reviews)}/{num_samples} accepted, "
                      f"{rejected_count} rejected, {total_attempts} total attempts")
            
            # Check if we're done
            if len(accepted_reviews) >= num_samples:
                print(f"✅ Target reached! Generated {len(accepted_reviews)}/{num_samples} reviews")
                break
            
            # Small delay between batches to avoid overwhelming APIs
            if len(accepted_reviews) < num_samples:
                await asyncio.sleep(0.5)
        
        # Calculate final stats
        if stats.total_generated > 0:
            stats.avg_time_per_review = stats.total_time_seconds / stats.total_generated
            stats.total_rejected = stats.total_generated - stats.total_accepted
        
        print(f"\n✓ Generation complete: {len(accepted_reviews)} accepted, "
              f"{stats.total_rejected} rejected, {total_attempts} total attempts")
        
        if len(accepted_reviews) < num_samples:
            print(f"⚠ Warning: Only generated {len(accepted_reviews)}/{num_samples} accepted reviews")
        
        return accepted_reviews
    
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
        """Select a tool name."""
        tool_names = [
            "ProjectFlow", "TeamSync", "DataViz Pro", "CloudConnect",
            "TaskMaster", "AnalyticsHub", "CommBridge", "WorkSpace Pro"
        ]
        return random.choice(tool_names)
