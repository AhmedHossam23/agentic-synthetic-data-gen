"""Model providers for OpenAI and Google."""

import os
from typing import Optional, Dict, Any
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try loading from current directory
        load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

from openai import OpenAI
import google.generativeai as genai

# Optional langsmith
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class ModelProvider:
    """Base class for model providers."""
    
    def __init__(self, name: str, model: str, temperature: float = 0.7, max_tokens: int = 500):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @traceable(name="generate_review")
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI model provider."""
    
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 500):
        super().__init__("openai", model, temperature, max_tokens)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Try loading .env again
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path, override=True)
                    api_key = os.getenv("OPENAI_API_KEY")
            except:
                pass
        
        if not api_key:
            env_path = Path(__file__).parent.parent.parent / ".env"
            error_msg = (
                "OPENAI_API_KEY not found in environment variables.\n"
                f"Please ensure:\n"
                f"1. .env file exists at: {env_path}\n"
                f"2. python-dotenv is installed: pip install python-dotenv\n"
                f"3. .env file contains: OPENAI_API_KEY=your-key-here\n"
                f"Or export it: export OPENAI_API_KEY='your-key'"
            )
            raise ValueError(error_msg)
        self.client = OpenAI(api_key=api_key)
    
    @traceable(name="openai_generate")
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at writing authentic SaaS tool reviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content


class GoogleProvider(ModelProvider):
    """Google Gemini model provider."""
    
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 500):
        super().__init__("google", model, temperature, max_tokens)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Try loading .env again
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path, override=True)
                    api_key = os.getenv("GOOGLE_API_KEY")
            except:
                pass
        
        if not api_key:
            env_path = Path(__file__).parent.parent.parent / ".env"
            error_msg = (
                "GOOGLE_API_KEY not found in environment variables.\n"
                f"Please ensure:\n"
                f"1. .env file exists at: {env_path}\n"
                f"2. python-dotenv is installed: pip install python-dotenv\n"
                f"3. .env file contains: GOOGLE_API_KEY=your-key-here\n"
                f"Or export it: export GOOGLE_API_KEY='your-key'"
            )
            raise ValueError(error_msg)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
    
    @traceable(name="google_generate")
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Google Gemini."""
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text


def create_provider(config: Dict[str, Any]) -> ModelProvider:
    """Factory function to create a model provider."""
    provider_type = config["provider"].lower()
    model = config["model"]
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 500)
    
    if provider_type == "openai":
        return OpenAIProvider(model, temperature, max_tokens)
    elif provider_type == "google":
        return GoogleProvider(model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
