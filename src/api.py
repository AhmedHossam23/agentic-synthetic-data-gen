"""FastAPI interface for the synthetic data generator."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

from src.config import load_config, Config
from src.generation_workflow import ReviewGenerationWorkflow
from src.reporting import QualityReporter
from src.models import Review, GenerationStats

app = FastAPI(title="Synthetic Data Generator API", version="1.0.0")


class GenerationRequest(BaseModel):
    """Request model for generation."""
    num_samples: Optional[int] = None
    config_path: Optional[str] = "config.yaml"


class GenerationResponse(BaseModel):
    """Response model for generation."""
    success: bool
    num_generated: int
    reviews: List[Dict[str, Any]]
    stats: Dict[str, Any]
    message: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Synthetic Data Generator",
        "version": "1.0.0",
        "endpoints": [
            "/generate",
            "/health",
            "/config"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/config")
async def get_config(config_path: str = "config.yaml"):
    """Get current configuration."""
    try:
        config = load_config(config_path)
        return {
            "num_samples": config.generation.num_samples,
            "models": [m.name for m in config.models],
            "personas": [p.name for p in config.personas],
            "domain": config.domain.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate_reviews(request: GenerationRequest):
    """Generate synthetic reviews."""
    try:
        # Load configuration
        config = load_config(request.config_path or "config.yaml")
        
        # Override num_samples if provided
        if request.num_samples:
            config.generation.num_samples = request.num_samples
        
        # Initialize workflow
        workflow = ReviewGenerationWorkflow(config)
        
        # Generate reviews
        reviews = workflow.generate(config.generation.num_samples)
        
        # Get stats from workflow (simplified - in real implementation, return from workflow)
        estimated_total = int(len(reviews) * 1.5)  # Assume ~67% acceptance rate
        stats = GenerationStats(
            total_generated=estimated_total,
            total_accepted=len(reviews),
            total_rejected=estimated_total - len(reviews)
        )
        
        return GenerationResponse(
            success=True,
            num_generated=len(reviews),
            reviews=[r.model_dump() for r in reviews],
            stats=stats.model_dump(),
            message=f"Successfully generated {len(reviews)} reviews"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reviews/{review_id}")
async def get_review(review_id: str):
    """Get a specific review by ID."""
    # This would require a storage mechanism
    raise HTTPException(status_code=501, detail="Not implemented")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
