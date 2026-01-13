"""Setup script for the project."""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Dependencies installed")


def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"⚠ Warning: Could not download NLTK data: {e}")


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = ["output", "data", "logs"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print("✓ Directories created")


def check_env_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠ Warning: .env file not found")
        print("Please create a .env file with:")
        print("  OPENAI_API_KEY=your_key")
        print("  GOOGLE_API_KEY=your_key")
        print("  LANGSMITH_API_KEY=your_key (optional)")
    else:
        print("✓ .env file found")


def main():
    """Run setup."""
    print("Setting up Synthetic Data Generator...\n")
    
    install_dependencies()
    download_nltk_data()
    create_directories()
    check_env_file()
    
    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("1. Configure your .env file with API keys")
    print("2. Review and adjust config.yaml if needed")
    print("3. Run: python main.py generate")


if __name__ == "__main__":
    main()
