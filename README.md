# Synthetic Data Generator for SaaS Tool Reviews

A professional-grade synthetic data generator for SaaS tool reviews with quality guardrails, built using LangGraph, DSPy, and LangSmith monitoring.

## Features

- **Multi-Model Support**: Generate reviews using OpenAI GPT-4 and Google Gemini
- **Quality Guardrails**: 
  - Diversity metrics (vocabulary overlap, semantic similarity)
  - Bias detection (sentiment skew, rating variance)
  - Domain realism validation
  - Automated rejection/regeneration
- **LangGraph Workflow**: Orchestrated generation pipeline with state management
- **DSPy Integration**: Structured generation and validation programs
- **LangSmith Monitoring**: Track quality and performance metrics
- **CLI & API**: Both command-line and REST API interfaces
- **Comprehensive Reporting**: Quality reports with metrics and comparisons

## Architecture

```
┌─────────────────┐
│   CLI / API     │
└────────┬────────┘
         │
┌────────▼────────┐
│  LangGraph      │
│   Workflow      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│ DSPy  │ │ Quality │
│Programs│ │Guardrails│
└───┬───┘ └──┬──────┘
    │         │
┌───▼─────────▼───┐
│ Model Providers │
│ (OpenAI/Google) │
└─────────────────┘
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd agentic-synthetic-data-gen
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
# Option 1: Use setup script (recommended)
python setup.py

# Option 2: Manual installation
pip install -r requirements.txt
```

**Note**: If you encounter dependency conflicts (especially with `sentence-transformers`), the code includes fallback mechanisms and will work without some optional dependencies. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for details.

4. **Download NLTK data** (required for text processing):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. **Set up environment variables**:
Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
LANGSMITH_API_KEY=your_langsmith_key  # Optional but recommended
LANGCHAIN_PROJECT=synthetic-data-generator
```

## Configuration

Edit `config.yaml` to customize:
- Number of samples to generate
- Model providers and weights
- Personas and their characteristics
- Quality guardrail thresholds
- Domain-specific terms

## Usage

### CLI

**Generate reviews**:
```bash
python main.py generate --num-samples 400 --output-dir output
```

**Validate configuration**:
```bash
python main.py validate
```

**Analyze generated reviews**:
```bash
python main.py analyze --reviews output/reviews.json
```

**Compare with real reviews**:
```bash
python main.py generate --real-reviews real_reviews.json
```

### API

**Start the API server**:
```bash
python -m src.api
```

**Generate reviews via API**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 100}'
```

## Output

The generator produces:
1. **reviews.json**: Generated reviews with metadata and quality scores
2. **quality_report.md**: Comprehensive quality report with:
   - Generation statistics
   - Model performance metrics
   - Quality metrics summary
   - Rating and persona distributions
   - Comparison with real reviews (if provided)
   - Recommendations

## Quality Guardrails

### Diversity Metrics
- **Vocabulary Overlap**: Ensures reviews don't repeat the same words excessively
- **Semantic Similarity**: Maintains appropriate semantic distance between reviews

### Bias Detection
- **Sentiment Skew**: Detects if sentiment distribution is unrealistic
- **Rating Variance**: Ensures rating distribution has appropriate variance

### Realism Validation
- **Readability**: Checks Flesch reading ease score
- **Domain Terms**: Validates use of domain-specific terminology
- **Length**: Ensures reviews are within expected length ranges

## Design Decisions

### Why LangGraph?
- **State Management**: Complex generation workflows require careful state tracking
- **Conditional Logic**: Quality guardrails need conditional acceptance/rejection paths
- **Retry Logic**: Failed generations need automatic retry mechanisms
- **Observability**: Built-in tracing and monitoring capabilities

### Why DSPy?
- **Structured Generation**: Ensures consistent output format
- **Validation**: Built-in validation programs for quality checks
- **Optimization**: Can be optimized for better quality over time
- **Modularity**: Separates generation logic from validation logic

### Why LangSmith?
- **Monitoring**: Track quality metrics over time
- **Debugging**: Identify issues in generation pipeline
- **Performance**: Monitor model performance and costs
- **Compliance**: Audit trail for generated data

## Trade-offs

1. **Quality vs Speed**: More quality checks = slower generation
   - **Solution**: Configurable thresholds, batch processing

2. **Diversity vs Realism**: Too diverse = unrealistic, too similar = low diversity
   - **Solution**: Balanced thresholds in config

3. **Model Costs**: Multiple models increase API costs
   - **Solution**: Weighted model selection, caching

4. **Rejection Rate**: High quality = high rejection rate
   - **Solution**: Configurable thresholds, retry logic

## Hardware/Model Limitations

- **API Rate Limits**: Both OpenAI and Google have rate limits
- **Token Costs**: Large-scale generation can be expensive
- **Processing Time**: Quality checks add computational overhead
- **Memory**: Sentence transformers require ~500MB RAM

## Example Output

```json
{
  "id": "uuid-here",
  "text": "As a CTO at a large enterprise, I've been using ProjectFlow for 6 months...",
  "rating": 4,
  "persona": "Enterprise CTO",
  "model_used": "openai",
  "quality_scores": {
    "diversity": 0.75,
    "bias": 0.25,
    "realism": 0.85,
    "overall": 0.68
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

## Acknowledgments

- LangGraph team for the workflow framework
- DSPy team for structured generation
- LangSmith for monitoring capabilities
