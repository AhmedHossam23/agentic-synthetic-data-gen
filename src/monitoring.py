"""LangSmith monitoring setup."""

import os
from langsmith import Client
from langsmith.run_helpers import traceable


def setup_langsmith():
    """Setup LangSmith monitoring."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    api_url = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
    
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = api_url
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "synthetic-data-generator")
        
        return Client(api_key=api_key, api_url=api_url)
    else:
        print("Warning: LANGSMITH_API_KEY not found. Monitoring will be limited.")
        return None


def get_traceable_decorator():
    """Get traceable decorator for LangSmith."""
    return traceable
