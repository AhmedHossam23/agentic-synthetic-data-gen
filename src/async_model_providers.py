"""Async model providers for OpenAI and Google."""

import os
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from openai import AsyncOpenAI
import google.generativeai as genai

# Optional langsmith
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class AsyncModelProvider:
    """Base class for async model providers."""
    
    def __init__(self, name: str, model: str, temperature: float = 0.7, max_tokens: int = 500):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @traceable(name="async_generate_review")
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt asynchronously."""
        raise NotImplementedError


class AsyncOpenAIProvider(AsyncModelProvider):
    """Async OpenAI model provider."""
    
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 500):
        super().__init__("openai", model, temperature, max_tokens)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
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
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=api_key)
    
    @traceable(name="async_openai_generate")
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI asynchronously."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at writing authentic SaaS tool reviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content


class AsyncGoogleProvider(AsyncModelProvider):
    """Async Google Gemini model provider."""
    
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 500):
        super().__init__("google", model, temperature, max_tokens)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
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
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
    
    @traceable(name="async_google_generate")
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Google Gemini asynchronously."""
        # Note: Google's genai library doesn't have native async support
        # We'll run it in a thread pool to avoid blocking
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # Run in executor to make it non-blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
        )
        return response.text


def create_async_provider(config: Dict[str, Any]) -> AsyncModelProvider:
    """Factory function to create an async model provider."""
    provider_type = config["provider"].lower()
    model = config["model"]
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 500)
    
    if provider_type == "openai":
        return AsyncOpenAIProvider(model, temperature, max_tokens)
    elif provider_type == "google":
        return AsyncGoogleProvider(model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
