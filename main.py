#!/usr/bin/env python3
"""Main entry point for the synthetic data generator."""

import sys
from pathlib import Path

# Load .env file first
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from {env_path}")
    else:
        load_dotenv()  # Try loading from current directory
except ImportError:
    pass  # dotenv not installed, will use system environment variables

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    try:
        import click
    except ImportError:
        missing.append("click")
    
    try:
        import pydantic
    except ImportError:
        missing.append("pydantic")
    
    if missing:
        print("❌ Missing required dependencies:", ", ".join(missing))
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        print("\nOr run the setup script:")
        print("  python setup.py")
        sys.exit(1)

def check_env_vars():
    """Check if required environment variables are set."""
    import os
    missing = []
    
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    
    if missing:
        print("⚠️  Warning: Missing environment variables:", ", ".join(missing))
        print("Please set them in your .env file or environment")
        print(f"Current .env path: {Path(__file__).parent / '.env'}")
        return False
    return True

if __name__ == "__main__":
    check_dependencies()
    check_env_vars()
    from src.cli import cli
    cli()
