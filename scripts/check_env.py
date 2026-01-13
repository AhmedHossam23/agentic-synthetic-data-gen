#!/usr/bin/env python3
"""Script to check environment variables."""

import os
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from: {env_path}")
    else:
        print(f"⚠ .env file not found at: {env_path}")
        load_dotenv()  # Try current directory
except ImportError:
    print("⚠ python-dotenv not installed. Using system environment variables only.")

print("\nEnvironment Variables Check:")
print("=" * 50)

# Check required variables
required_vars = {
    "OPENAI_API_KEY": "OpenAI API key",
    "GOOGLE_API_KEY": "Google API key",
    "LANGSMITH_API_KEY": "LangSmith API key (optional)",
}

for var_name, description in required_vars.items():
    value = os.getenv(var_name)
    if value:
        # Show first and last 4 characters for security
        masked = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
        print(f"✓ {var_name}: {masked} ({description})")
    else:
        status = "⚠" if var_name != "LANGSMITH_API_KEY" else "○"
        print(f"{status} {var_name}: NOT SET ({description})")

print("\n" + "=" * 50)
print("\nTo set variables:")
print("1. Edit .env file in project root")
print("2. Or export them in your shell:")
print("   export OPENAI_API_KEY='your-key'")
print("   export GOOGLE_API_KEY='your-key'")
