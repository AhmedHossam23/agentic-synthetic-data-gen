# System Architecture

## Architecture Diagram

```mermaid
graph TB
    subgraph "User Interfaces"
        CLI[CLI Interface<br/>Click Framework]
        API[FastAPI REST API<br/>Uvicorn Server]
    end

    subgraph "Application Layer"
        CONFIG[Configuration Manager<br/>YAML Config Parser]
        WORKFLOW_SYNC[LangGraph Workflow<br/>Synchronous Mode]
        WORKFLOW_ASYNC[Async Review Generator<br/>Parallel Processing]
    end

    subgraph "Generation Engine"
        PERSONA[Persona Selector<br/>Role-based Sampling]
        MODEL_SELECT[Model Selector<br/>Weighted Distribution]
        PROMPT_BUILDER[Prompt Builder<br/>Template Engine]
        DSPY[DSPy Programs<br/>Structured Generation<br/>Optional]
    end

    subgraph "Model Providers"
        OPENAI_PROV[OpenAI Provider<br/>GPT-4 Turbo]
        GOOGLE_PROV[Google Provider<br/>Gemini 2.0 Flash]
        ASYNC_OPENAI[Async OpenAI<br/>AsyncOpenAI Client]
        ASYNC_GOOGLE[Async Google<br/>Async Gemini Client]
    end

    subgraph "Quality Assurance"
        GUARDRAIL[Quality Guardrails<br/>Multi-metric Evaluation]
        DIVERSITY[Diversity Metrics<br/>Vocabulary Overlap<br/>Semantic Similarity]
        BIAS[Bias Detection<br/>Sentiment Skew<br/>Rating Variance]
        REALISM[Realism Validation<br/>Readability<br/>Domain Terms<br/>Length Check]
        EMBEDDINGS[Gemini Embeddings<br/>Semantic Similarity]
    end

    subgraph "Monitoring & Observability"
        LANGSMITH[LangSmith Client<br/>Tracing & Monitoring]
        TRACEABLE[Traceable Decorators<br/>@traceable]
    end

    subgraph "External Services"
        OPENAI_API[OpenAI API<br/>api.openai.com]
        GOOGLE_API[Google Gemini API<br/>generativelanguage.googleapis.com]
        LANGSMITH_API[LangSmith API<br/>api.smith.langchain.com]
    end

    subgraph "Data Processing"
        REPORTER[Quality Reporter<br/>Statistics & Analysis]
        JSON_OUT[Output Handler<br/>JSON Export]
        REPORT_OUT[Report Generator<br/>Markdown Reports]
    end

    subgraph "Storage & Output"
        REVIEWS_JSON[reviews.json<br/>Generated Reviews]
        QUALITY_REPORT[quality_report.md<br/>Quality Metrics]
    end

    CLI --> CONFIG
    API --> CONFIG
    CONFIG --> WORKFLOW_SYNC
    CONFIG --> WORKFLOW_ASYNC

    WORKFLOW_SYNC --> PERSONA
    WORKFLOW_SYNC --> MODEL_SELECT
    WORKFLOW_SYNC --> PROMPT_BUILDER
    WORKFLOW_SYNC --> OPENAI_PROV
    WORKFLOW_SYNC --> GOOGLE_PROV
    WORKFLOW_SYNC --> GUARDRAIL
    WORKFLOW_SYNC --> LANGSMITH

    WORKFLOW_ASYNC --> PERSONA
    WORKFLOW_ASYNC --> MODEL_SELECT
    WORKFLOW_ASYNC --> PROMPT_BUILDER
    WORKFLOW_ASYNC --> ASYNC_OPENAI
    WORKFLOW_ASYNC --> ASYNC_GOOGLE
    WORKFLOW_ASYNC --> GUARDRAIL
    WORKFLOW_ASYNC --> LANGSMITH

    PERSONA --> PROMPT_BUILDER
    PROMPT_BUILDER --> DSPY
    DSPY --> OPENAI_PROV
    DSPY --> GOOGLE_PROV

    OPENAI_PROV --> OPENAI_API
    GOOGLE_PROV --> GOOGLE_API
    ASYNC_OPENAI --> OPENAI_API
    ASYNC_GOOGLE --> GOOGLE_API

    GUARDRAIL --> DIVERSITY
    GUARDRAIL --> BIAS
    GUARDRAIL --> REALISM
    DIVERSITY --> EMBEDDINGS
    REALISM --> EMBEDDINGS
    EMBEDDINGS --> GOOGLE_API

    TRACEABLE --> LANGSMITH
    LANGSMITH --> LANGSMITH_API

    WORKFLOW_SYNC --> REPORTER
    WORKFLOW_ASYNC --> REPORTER
    REPORTER --> JSON_OUT
    REPORTER --> REPORT_OUT
    JSON_OUT --> REVIEWS_JSON
    REPORT_OUT --> QUALITY_REPORT

```

## Technology Stack

```mermaid
graph LR
    subgraph "Core Framework"
        PYTHON[Python 3.8+]
        LANGGRAPH[LangGraph<br/>Workflow Orchestration]
        DSPY[DSPy<br/>Structured Generation]
    end

    subgraph "LLM Providers"
        OPENAI_SDK[OpenAI SDK<br/>v1.12.0+]
        GOOGLE_SDK[Google Generative AI<br/>v0.3.0+]
    end

    subgraph "Web Framework"
        FASTAPI[FastAPI<br/>REST API]
        UVICORN[Uvicorn<br/>ASGI Server]
    end

    subgraph "CLI Framework"
        CLICK[Click<br/>Command Line Interface]
        RICH[Rich<br/>Terminal UI]
    end

    subgraph "Data Processing"
        PANDAS[Pandas<br/>Data Analysis]
        NUMPY[NumPy<br/>Numerical Computing]
        PYDANTIC[Pydantic<br/>Data Validation]
    end

    subgraph "NLP & Quality"
        NLTK[NLTK<br/>Text Processing]
        TEXTSTAT[TextStat<br/>Readability Metrics]
        GEMINI_EMB[Gemini Embeddings<br/>Semantic Similarity]
    end

    subgraph "Monitoring"
        LANGSMITH_SDK[LangSmith SDK<br/>Observability]
    end

    subgraph "Configuration"
        YAML[PyYAML<br/>Config Parsing]
        DOTENV[python-dotenv<br/>Environment Variables]
    end

    subgraph "Utilities"
        TENACITY[Tenacity<br/>Retry Logic]
        TQDM[tqdm<br/>Progress Bars]
        ASYNCIO[asyncio<br/>Async Processing]
    end

    PYTHON --> LANGGRAPH
    PYTHON --> DSPY
    PYTHON --> FASTAPI
    PYTHON --> CLICK
    LANGGRAPH --> LANGSMITH_SDK
    OPENAI_SDK --> PYTHON
    GOOGLE_SDK --> PYTHON
    FASTAPI --> UVICORN
    CLICK --> RICH
    PANDAS --> NUMPY
    NLTK --> PYTHON
    TEXTSTAT --> PYTHON
    GEMINI_EMB --> GOOGLE_SDK
    LANGSMITH_SDK --> PYTHON
    YAML --> PYTHON
    DOTENV --> PYTHON
    TENACITY --> PYTHON
    TQDM --> PYTHON
    ASYNCIO --> PYTHON

    classDef core fill:#4caf50,stroke:#2e7d32,stroke-width:2px,color:#fff
    classDef llm fill:#2196f3,stroke:#1565c0,stroke-width:2px,color:#fff
    classDef web fill:#ff9800,stroke:#e65100,stroke-width:2px,color:#fff
    classDef cli fill:#9c27b0,stroke:#6a1b9a,stroke-width:2px,color:#fff
    classDef data fill:#00bcd4,stroke:#00838f,stroke-width:2px,color:#fff
    classDef nlp fill:#f44336,stroke:#c62828,stroke-width:2px,color:#fff
    classDef monitor fill:#ffc107,stroke:#f57c00,stroke-width:2px,color:#000
    classDef config fill:#795548,stroke:#5d4037,stroke-width:2px,color:#fff
    classDef util fill:#607d8b,stroke:#455a64,stroke-width:2px,color:#fff

    class PYTHON,LANGGRAPH,DSPY core
    class OPENAI_SDK,GOOGLE_SDK llm
    class FASTAPI,UVICORN web
    class CLICK,RICH cli
    class PANDAS,NUMPY,PYDANTIC data
    class NLTK,TEXTSTAT,GEMINI_EMB nlp
    class LANGSMITH_SDK monitor
    class YAML,DOTENV config
    class TENACITY,TQDM,ASYNCIO util
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI_API
    participant Config
    participant Workflow
    participant Persona
    participant Model
    participant LLM_API
    participant Guardrail
    participant Embeddings
    participant Reporter
    participant Output

    User->>CLI_API: Generate Reviews Request
    CLI_API->>Config: Load Configuration
    Config-->>CLI_API: Config Object
    
    loop For each review
        CLI_API->>Workflow: Generate Review
        Workflow->>Persona: Select Persona & Rating
        Persona-->>Workflow: Persona + Rating
        
        Workflow->>Model: Select Model Provider
        Model-->>Workflow: Provider Instance
        
        Workflow->>Model: Build Prompt
        Model-->>Workflow: Prompt String
        
        Workflow->>LLM_API: Generate Text
        LLM_API-->>Workflow: Review Text
        
        Workflow->>Guardrail: Evaluate Quality
        Guardrail->>Embeddings: Calculate Similarity
        Embeddings-->>Guardrail: Similarity Score
        Guardrail->>Guardrail: Check Diversity
        Guardrail->>Guardrail: Check Bias
        Guardrail->>Guardrail: Check Realism
        Guardrail-->>Workflow: Quality Metrics
        
        alt Quality Passes
            Workflow->>Workflow: Accept Review
        else Quality Fails
            Workflow->>Workflow: Reject & Retry
        end
    end
    
    Workflow->>Reporter: Generate Report
    Reporter->>Reporter: Calculate Statistics
    Reporter->>Output: Save Reviews JSON
    Reporter->>Output: Save Quality Report
    Output-->>User: reviews.json + quality_report.md
```

## Component Interaction Diagram

```mermaid
graph TD
    subgraph "Generation Pipeline"
        A[Start] --> B{Async Mode?}
        B -->|Yes| C[Async Generator]
        B -->|No| D[Sync Workflow]
        
        C --> E[Batch Generation]
        D --> F[Sequential Generation]
        
        E --> G[Parallel API Calls]
        F --> H[Single API Call]
        
        G --> I[Rate Limiting<br/>Semaphore]
        H --> I
        
        I --> J[Model Provider]
        J --> K{Provider Type}
        K -->|OpenAI| L[GPT-4 Turbo]
        K -->|Google| M[Gemini 2.0]
        
        L --> N[Review Text]
        M --> N
        
        N --> O[Quality Evaluation]
        O --> P{Diversity Check}
        O --> Q{Bias Check}
        O --> R{Realism Check}
        
        P --> S{Pass?}
        Q --> S
        R --> S
        
        S -->|Yes| T[Accept Review]
        S -->|No| U{Retries Left?}
        U -->|Yes| J
        U -->|No| V[Reject Review]
        
        T --> W{Enough Reviews?}
        V --> W
        W -->|No| B
        W -->|Yes| X[Generate Report]
    end
    
    X --> Y[Save Output]
    Y --> Z[End]
    
    classDef startEnd fill:#4caf50,stroke:#2e7d32,stroke-width:2px,color:#fff
    classDef process fill:#2196f3,stroke:#1565c0,stroke-width:2px,color:#fff
    classDef decision fill:#ff9800,stroke:#e65100,stroke-width:2px,color:#fff
    classDef quality fill:#f44336,stroke:#c62828,stroke-width:2px,color:#fff
    
    class A,Z startEnd
    class C,D,E,F,G,H,I,J,L,M,N,O,X,Y process
    class B,K,P,Q,S,U,W decision
    class O,P,Q,R quality
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Developer Machine<br/>Python 3.8+]
        VENV[Virtual Environment<br/>venv]
        ENV[.env File<br/>API Keys]
    end

    subgraph "Application Runtime"
        APP[Python Application<br/>CLI or API Server]
        MEM[Memory<br/>Review Cache<br/>Embedding Cache]
    end

    subgraph "External APIs"
        EXT_OPENAI[OpenAI API<br/>Cloud Service]
        EXT_GOOGLE[Google Gemini API<br/>Cloud Service]
        EXT_LANGSMITH[LangSmith API<br/>Monitoring Service]
    end

    subgraph "Output Filesystem"
        FS[Local Filesystem<br/>output/]
        JSON[reviews.json<br/>Generated Data]
        MD[quality_report.md<br/>Analysis Report]
    end

    DEV --> VENV
    VENV --> APP
    ENV --> APP
    APP --> MEM
    APP --> EXT_OPENAI
    APP --> EXT_GOOGLE
    APP --> EXT_LANGSMITH
    APP --> FS
    FS --> JSON
    FS --> MD

    classDef dev fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef runtime fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef external fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef storage fill:#e8f5e9,stroke:#388e3c,stroke-width:2px

    class DEV,VENV,ENV dev
    class APP,MEM runtime
    class EXT_OPENAI,EXT_GOOGLE,EXT_LANGSMITH external
    class FS,JSON,MD storage
```

## Key Design Patterns

1. **Factory Pattern**: Model provider creation (`create_provider`, `create_async_provider`)
2. **Strategy Pattern**: Multiple model providers with unified interface
3. **Observer Pattern**: LangSmith tracing for monitoring
4. **Template Method**: Workflow orchestration in LangGraph
5. **Decorator Pattern**: `@traceable` for observability
6. **Retry Pattern**: Tenacity for resilient API calls
7. **Semaphore Pattern**: Rate limiting in async generation

## Performance Characteristics

- **Synchronous Mode**: Sequential generation, ~1-2 reviews/second
- **Async Mode**: Parallel batch processing, ~10-20 reviews/second (batch_size dependent)
- **Quality Checks**: Add ~0.5-1 second per review (embedding calculation)
- **Rate Limiting**: Configurable via semaphore (default: batch_size=10)
- **Caching**: Embedding cache reduces redundant API calls
