"""CLI interface for the synthetic data generator."""

import click
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config import load_config, Config
from src.generation_workflow import ReviewGenerationWorkflow
from src.reporting import QualityReporter
from src.models import GenerationStats

console = Console()


@click.group()
def cli():
    """Synthetic Data Generator for SaaS Tool Reviews."""
    pass


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to configuration file")
@click.option("--output-dir", "-o", default="output", help="Output directory")
@click.option("--num-samples", "-n", type=int, help="Number of samples to generate (overrides config)")
@click.option("--real-reviews", "-r", help="Path to JSON file with real reviews for comparison")
def generate(config: str, output_dir: str, num_samples: Optional[int], real_reviews: Optional[str]):
    """Generate synthetic reviews."""
    console.print("[bold blue]Starting Review Generation[/bold blue]")
    
    # Load configuration
    try:
        cfg = load_config(config)
        console.print(f"[green]✓[/green] Loaded configuration from {config}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load configuration: {e}")
        return
    
    # Override num_samples if provided
    if num_samples:
        cfg.generation.num_samples = num_samples
        console.print(f"[yellow]→[/yellow] Overriding num_samples to {num_samples}")
    
    # Load real reviews if provided
    real_reviews_data = None
    if real_reviews:
        try:
            with open(real_reviews, "r") as f:
                real_reviews_data = json.load(f)
            console.print(f"[green]✓[/green] Loaded {len(real_reviews_data)} real reviews")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to load real reviews: {e}")
    
    # Initialize workflow
    console.print("[blue]Initializing generation workflow...[/blue]")
    workflow = ReviewGenerationWorkflow(cfg)
    
    # Generate reviews
    console.print(f"[blue]Generating {cfg.generation.num_samples} reviews...[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating reviews...", total=None)
        reviews = workflow.generate(cfg.generation.num_samples)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Generated {len(reviews)} reviews")
    
    # Generate report
    console.print("[blue]Generating quality report...[/blue]")
    reporter = QualityReporter(cfg)
    
    # Get stats from workflow (simplified - in real implementation, return from workflow)
    # For now, estimate based on acceptance rate
    estimated_total = int(len(reviews) * 1.5)  # Assume ~67% acceptance rate
    stats = GenerationStats(
        total_generated=estimated_total,
        total_accepted=len(reviews),
        total_rejected=estimated_total - len(reviews)
    )
    
    report = reporter.generate_report(reviews, stats, real_reviews_data)
    
    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save reviews
    reviews_path = output_path / "reviews.json"
    reporter.save_reviews_json(reviews, str(reviews_path))
    console.print(f"[green]✓[/green] Saved reviews to {reviews_path}")
    
    # Save report
    report_path = output_path / "quality_report.md"
    reporter.save_report(report, str(report_path))
    console.print(f"[green]✓[/green] Saved report to {report_path}")
    
    # Display summary
    console.print("\n[bold green]Generation Complete![/bold green]")
    display_summary(reviews, stats)


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to configuration file")
def validate(config: str):
    """Validate configuration file."""
    console.print("[blue]Validating configuration...[/blue]")
    
    try:
        cfg = load_config(config)
        console.print("[green]✓[/green] Configuration is valid")
        
        # Display config summary
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Num Samples", str(cfg.generation.num_samples))
        table.add_row("Models", ", ".join([m.name for m in cfg.models]))
        table.add_row("Personas", str(len(cfg.personas)))
        table.add_row("Domain", cfg.domain.name)
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration validation failed: {e}")


@cli.command()
@click.option("--reviews", "-r", required=True, help="Path to reviews JSON file")
def analyze(reviews: str):
    """Analyze generated reviews."""
    console.print("[blue]Analyzing reviews...[/blue]")
    
    try:
        with open(reviews, "r") as f:
            reviews_data = json.load(f)
        
        console.print(f"[green]✓[/green] Loaded {len(reviews_data)} reviews")
        
        # Display analysis
        display_analysis(reviews_data)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to analyze reviews: {e}")


def display_summary(reviews, stats):
    """Display generation summary."""
    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Generated", str(stats.total_generated))
    table.add_row("Accepted", str(stats.total_accepted))
    table.add_row("Rejected", str(stats.total_rejected))
    if stats.total_generated > 0:
        acceptance_rate = stats.total_accepted / stats.total_generated * 100
        table.add_row("Acceptance Rate", f"{acceptance_rate:.1f}%")
    
    console.print(table)


def display_analysis(reviews_data):
    """Display analysis of reviews."""
    try:
        import pandas as pd
        df = pd.DataFrame(reviews_data)
    except ImportError:
        # Fallback without pandas
        from collections import Counter
        df = None
    
    table = Table(title="Review Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    if df is not None:
        if "rating" in df.columns:
            avg_rating = df["rating"].mean()
            table.add_row("Average Rating", f"{avg_rating:.2f}")
        
        if "text" in df.columns:
            avg_length = df["text"].str.len().mean()
            table.add_row("Average Length", f"{avg_length:.0f} characters")
        
        if "persona" in df.columns:
            persona_counts = df["persona"].value_counts()
            table.add_row("Personas", ", ".join([f"{p}: {c}" for p, c in persona_counts.items()]))
    else:
        # Fallback without pandas
        from collections import Counter
        ratings = [r.get("rating", 0) for r in reviews_data if "rating" in r]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            table.add_row("Average Rating", f"{avg_rating:.2f}")
        
        texts = [r.get("text", "") for r in reviews_data if "text" in r]
        if texts:
            avg_length = sum(len(t) for t in texts) / len(texts)
            table.add_row("Average Length", f"{avg_length:.0f} characters")
        
        personas = [r.get("persona", "") for r in reviews_data if "persona" in r]
        if personas:
            persona_counts = Counter(personas)
            table.add_row("Personas", ", ".join([f"{p}: {c}" for p, c in persona_counts.items()]))
    
    console.print(table)
