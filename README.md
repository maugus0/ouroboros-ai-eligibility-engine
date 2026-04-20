# Ouroboros Eligibility Engine

Microservice for the **Ouroboros AI** scholarship discovery platform. The Eligibility Engine computes rule-based + LLM-assisted match scores for programs and scholarships, generates explainable attribution reports, and leverages PostgreSQL + pgvector for semantic research alignment matching. Raw SQL with `asyncpg` and the repository pattern (NO SQLAlchemy).

**Repository:** <https://github.com/maugus0/ouroboros-ai-eligibility-engine>

***

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)
- [Integration Contract](#integration-contract)
- [Scoring System](#scoring-system)
- [Prompt System](#prompt-system)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Attribution](#attribution)

***

## Overview

The Eligibility Engine is a critical microservice in the Ouroboros AI platform that:

1. **Computes program match scores** with weighted scoring (GPA 20%, Relevance 30%, Prerequisites 25%, Research Alignment 15%, Practical 10%)
2. **Computes scholarship match scores** with weighted scoring (Eligibility 40%, Preferred Criteria 30%, Funding 20%, Competition 10%)
3. **Generates explainable attribution reports** with strengths, gaps, reasoning, and confidence levels
4. **Performs semantic matching** via pgvector for research interest similarity
5. **Stores historical results** for audit and analysis

**Key Design Principles:**

- No direct frontend access — only the orchestrator calls this service via `X-Service-Token`
- No ORM overhead — raw SQL with `asyncpg` async connection pool
- PostgreSQL + pgvector — single database for relational + vector data
- Rule-based deterministic scoring + LLM semantic reasoning
- Explainability-first design — every score includes breakdown and reasoning
- LLM resilience — OpenAI primary, Anthropic fallback with retry logic
- JSON prompt templates — version-controlled in `prompts/` with runtime context injection

***

## Architecture

```
┌─────────────────────────────────────┐
│    Orchestrator Service (8000)      │
│    (Single Entry Point)             │
└──────────────┬──────────────────────┘
               │
               │ X-Service-Token
               ▼
┌──────────────────────────────────────────────────────┐
│         Eligibility Engine (8004)                    │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │  API Layer (FastAPI)                        │     │
│  │  POST /matching/evaluate                     │     │
│  │  GET  /matching/results/{user_id}           │     │
│  │  GET  /matching/results/detail/{match_id}   │     │
│  │  GET  /attribution/report/{match_id}        │     │
│  └─────────────────────┬───────────────────────┘     │
│                        │                             │
│  ┌─────────────────────▼───────────────────────┐     │
│  │  Service Layer                              │     │
│  │  MatchingService    (orchestrates pipeline)  │     │
│  │  ProgramScorer      (rule-based scoring)     │     │
│  │  ScholarshipScorer  (rule-based scoring)     │     │
│  │  ExplainabilityService (attribution reports) │     │
│  │  EmbeddingService   (pgvector similarity)    │     │
│  │  LLMPipelineService (OpenAI → Anthropic)     │     │
│  └─────────────────────┬───────────────────────┘     │
│                        │                             │
│  ┌─────────────────────▼───────────────────────┐     │
│  │  Repository Layer (Raw SQL)                 │     │
│  │  MatchResultRepository                      │     │
│  │  AttributionReportRepository                │     │
│  │  ResearchEmbeddingRepository (pgvector)      │     │
│  │  ScoringHistoryRepository                   │     │
│  └─────────────────────────────────────────────┘     │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
      ┌─────────────────┐
      │ PostgreSQL 16   │
      │ + pgvector      │
      │ (asyncpg)       │
      └─────────────────┘
```

***

## Features

### Program Match Scoring

- **GPA (20%)** — Proportional scoring against minimum requirement
- **Relevance (30%)** — Keyword overlap between student skills and program keywords
- **Prerequisites (25%)** — Course completion ratio check
- **Research Alignment (15%)** — Cosine similarity via pgvector embeddings
- **Practical Factors (10%)** — Location preference, tuition affordability

### Scholarship Match Scoring

- **Eligibility (40%)** — Hard criteria: GPA, citizenship, field of study, degree level
- **Preferred Criteria (30%)** — Soft criteria: activities, achievements, leadership
- **Funding Coverage (20%)** — Award amount vs. student financial need
- **Competition Estimate (10%)** — Acceptance rate or applicant pool heuristic

### Explainability / Attribution Reports

- Strengths and gaps extracted from score breakdown
- LLM-generated reasoning narrative (with rule-based fallback)
- Actionable recommendations with priority levels
- Confidence levels (high / medium / low)

### Semantic Research Matching (pgvector)

- Text embeddings via OpenAI `text-embedding-3-small` (1536 dimensions)
- Cosine similarity search with IVFFlat indexing
- Configurable top-k and similarity threshold

***

## Prerequisites

| Tool              | Version | Purpose                                          |
| ----------------- | ------- | ------------------------------------------------ |
| Python            | 3.11+   | Runtime                                          |
| PostgreSQL        | 16+     | Database (with pgvector extension)               |
| OpenAI API Key    | —       | Primary LLM + embeddings provider                |
| Anthropic API Key | —       | Fallback LLM provider (optional but recommended) |
| Docker            | 24.0+   | Containerised deployment (optional)              |

***

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/maugus0/ouroboros-ai-eligibility-engine.git
cd ouroboros-ai-eligibility-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
DB_HOST=localhost
DB_NAME=ouroboros_eligibility_db
DB_USERNAME=postgres
DB_PASSWORD=your_postgres_password

X_SERVICE_TOKEN=your-secret-service-token-change-this

OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

See [Configuration](#configuration) for the full reference.

### 3. Database Setup

**Option A: Docker (Recommended)**

```bash
docker compose up postgres -d
docker compose logs -f postgres   # wait for "ready to accept connections"
```

**Option B: Local PostgreSQL**

```bash
psql -U postgres -c "CREATE DATABASE ouroboros_eligibility_db;"
psql -U postgres -d ouroboros_eligibility_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 4. Run Migrations

```bash
python scripts/run_migrations.py
```

### 5. Seed Test Embeddings (Optional)

```bash
python scripts/seed_test_embeddings.py
```

### 6. Start the Service

```bash
./start.sh
# or: uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

### 7. Verify Health

```bash
curl http://localhost:8004/health
# {"status":"healthy","version":"0.1.0","database":"connected"}
```

Swagger docs are available at `http://localhost:8004/docs`.

***

## Configuration

### Environment Variables

| Variable                         | Required    | Default                    | Description                                         |
| -------------------------------- | ----------- | -------------------------- | --------------------------------------------------- |
| **Database**                     | <br />      | <br />                     | <br />                                              |
| `DB_HOST`                        | No          | `localhost`                | PostgreSQL host                                     |
| `DB_PORT`                        | No          | `5432`                     | PostgreSQL port                                     |
| `DB_NAME`                        | No          | `ouroboros_eligibility_db` | Database name                                       |
| `DB_USERNAME`                    | No          | `postgres`                 | PostgreSQL user                                     |
| `DB_PASSWORD`                    | Yes         | —                          | PostgreSQL password                                 |
| `DB_POOL_MIN_SIZE`               | No          | `5`                        | Min connections in pool                             |
| `DB_POOL_MAX_SIZE`               | No          | `20`                       | Max connections in pool                             |
| **Service Auth**                 | <br />      | <br />                     | <br />                                              |
| `X_SERVICE_TOKEN`                | Yes         | —                          | Inter-service auth token (shared with orchestrator) |
| **LLM — OpenAI**                 | <br />      | <br />                     | <br />                                              |
| `OPENAI_API_KEY`                 | Yes         | —                          | OpenAI API key                                      |
| `OPENAI_MODEL`                   | No          | `gpt-4o-mini`              | Chat model identifier                               |
| `OPENAI_EMBEDDING_MODEL`         | No          | `text-embedding-3-small`   | Embedding model                                     |
| `OPENAI_MAX_TOKENS`              | No          | `1500`                     | Max output tokens                                   |
| `OPENAI_TEMPERATURE`             | No          | `0.1`                      | Sampling temperature                                |
| **LLM — Anthropic**              | <br />      | <br />                     | <br />                                              |
| `ANTHROPIC_API_KEY`              | Recommended | —                          | Anthropic API key (fallback)                        |
| `ANTHROPIC_MODEL`                | No          | `claude-sonnet-4-20250514` | Model identifier                                    |
| `ANTHROPIC_MAX_TOKENS`           | No          | `1500`                     | Max output tokens                                   |
| **Scoring**                      | <br />      | <br />                     | <br />                                              |
| `PROGRAM_WEIGHT_GPA`             | No          | `20`                       | GPA weight (sum must = 100)                         |
| `PROGRAM_WEIGHT_RELEVANCE`       | No          | `30`                       | Relevance weight                                    |
| `PROGRAM_WEIGHT_PREREQUISITES`   | No          | `25`                       | Prerequisites weight                                |
| `PROGRAM_WEIGHT_RESEARCH`        | No          | `15`                       | Research alignment weight                           |
| `PROGRAM_WEIGHT_PRACTICAL`       | No          | `10`                       | Practical factors weight                            |
| `SCHOLARSHIP_WEIGHT_ELIGIBILITY` | No          | `40`                       | Eligibility weight                                  |
| `SCHOLARSHIP_WEIGHT_PREFERRED`   | No          | `30`                       | Preferred criteria weight                           |
| `SCHOLARSHIP_WEIGHT_FUNDING`     | No          | `20`                       | Funding coverage weight                             |
| `SCHOLARSHIP_WEIGHT_COMPETITION` | No          | `10`                       | Competition estimate weight                         |
| `MATCH_SCORE_THRESHOLD`          | No          | `50.0`                     | Minimum viable match                                |
| `HIGH_CONFIDENCE_THRESHOLD`      | No          | `75.0`                     | High confidence threshold                           |
| **Vector Search**                | <br />      | <br />                     | <br />                                              |
| `EMBEDDING_DIMENSION`            | No          | `1536`                     | Embedding vector dimension                          |
| `VECTOR_SIMILARITY_TOP_K`        | No          | `5`                        | Max similar results                                 |
| `VECTOR_SIMILARITY_THRESHOLD`    | No          | `0.7`                      | Minimum cosine similarity                           |
| **Application**                  | <br />      | <br />                     | <br />                                              |
| `LOG_LEVEL`                      | No          | `INFO`                     | `DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL`             |
| `USE_MOCK_DATA`                  | No          | `false`                    | Use in-memory repos (tests only)                    |
| `ALLOW_DB_FAILURE`               | No          | `false`                    | Continue if DB unavailable (tests only)             |

### Docker / CI Prefix Compatibility

The service also reads `POSTGRES_*` variables for Docker/CI environments. Resolution logic lives in the `settings.get_db_*()` helpers in `app/config.py`.

***

## Database Schema

### Tables

| Table                 | Purpose                                                           |
| --------------------- | ----------------------------------------------------------------- |
| `match_results`       | Program and scholarship match scores with JSONB breakdown         |
| `attribution_reports` | Explainability data (strengths, gaps, reasoning, recommendations) |
| `scoring_history`     | Audit trail for all scoring operations                            |
| `research_embeddings` | Vector storage for semantic research alignment (pgvector)         |
| `llm_call_logs`       | Audit trail for all LLM API calls                                 |

### Migrations

Run in order via `python scripts/run_migrations.py`:

```
migrations/
├── 001_enable_pgvector.sql
├── 002_create_match_results.sql
├── 003_create_attribution_reports.sql
├── 004_create_scoring_history.sql
├── 005_create_research_embeddings.sql
└── 006_create_llm_call_logs.sql
```

***

## API Endpoints

**Base URL**: `http://localhost:8004`

All endpoints (except health) require the `X-Service-Token` header.

### Health

| Method | Path      | Auth | Description            |
| ------ | --------- | ---- | ---------------------- |
| GET    | `/`       | No   | Root health check      |
| GET    | `/health` | No   | Detailed health status |

### Matching

| Method | Path                                  | Description                                             |
| ------ | ------------------------------------- | ------------------------------------------------------- |
| POST   | `/matching/evaluate`                  | Evaluate a student-program or student-scholarship match |
| GET    | `/matching/results/{user_id}`         | Get paginated match results for a user                  |
| GET    | `/matching/results/detail/{match_id}` | Get a single match result by ID                         |

### Attribution

| Method | Path                             | Description                            |
| ------ | -------------------------------- | -------------------------------------- |
| GET    | `/attribution/report/{match_id}` | Get the attribution report for a match |

***

## Integration Contract

The Eligibility Engine is designed to be called only by the orchestrator or other trusted backend services.

### Authentication

- All non-health endpoints require `X-Service-Token`
- The value must match the shared secret configured in both the orchestrator and the eligibility-engine
- Frontend clients must not call this service directly

### Caller Requirements

- `user_id` must be a UUID string because result persistence uses PostgreSQL UUID columns
- `entity_type` must be either `program` or `scholarship`
- `entity_data.entity_type` should match the top-level `entity_type`
- `include_attribution=false` is supported for fast result-only flows

### Required Proxy Behaviour

When the orchestrator proxies requests into this service it should:

- inject the authenticated user ID as `user_id`
- forward `X-Trace-ID` for log correlation
- map downstream availability errors to orchestrator-facing `502/504`
- keep `success/message/data` response semantics intact

### Suggested Smoke Validation

```bash
curl http://localhost:8004/health

curl -X POST http://localhost:8004/matching/evaluate \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: $X_SERVICE_TOKEN" \
  -d '{
    "user_id": "11111111-1111-1111-1111-111111111111",
    "entity_type": "program",
    "entity_id": "program-demo-1",
    "user_profile": {
      "gpa_normalized": 3.8,
      "major": "Computer Science",
      "technical_skills": ["Python", "Machine Learning", "AI"],
      "completed_courses": ["Algorithms", "Machine Learning"],
      "research_interests": "Natural language processing and multilingual AI systems",
      "preferred_locations": ["Sydney"],
      "budget_usd": 40000
    },
    "entity_data": {
      "entity_type": "program",
      "minimum_gpa": 3.5,
      "keywords": ["Machine Learning", "AI", "NLP"],
      "prerequisites": ["Algorithms", "Machine Learning"],
      "location": "Sydney",
      "tuition_usd": 35000
    },
    "include_attribution": false
  }'
```

***

## Scoring System

### Program Scoring Weights (configurable, must sum to 100)

| Component          | Weight | Method                                           |
| ------------------ | ------ | ------------------------------------------------ |
| GPA                | 20%    | Proportional scoring against minimum requirement |
| Relevance          | 30%    | Keyword overlap (skills vs program keywords)     |
| Prerequisites      | 25%    | Course completion ratio                          |
| Research Alignment | 15%    | Cosine similarity via pgvector                   |
| Practical Factors  | 10%    | Location preference + budget affordability       |

### Scholarship Scoring Weights (configurable, must sum to 100)

| Component            | Weight | Method                                      |
| -------------------- | ------ | ------------------------------------------- |
| Eligibility          | 40%    | Binary pass/fail per hard criterion         |
| Preferred Criteria   | 30%    | Activity/achievement overlap                |
| Funding Coverage     | 20%    | Award amount / financial need ratio         |
| Competition Estimate | 10%    | Acceptance rate or applicant pool heuristic |

### Confidence Levels

| Score Range | Confidence |
| ----------- | ---------- |
| 75+         | High       |
| 50-74.9     | Medium     |
| < 50        | Low        |

***

## Prompt System

Prompts are stored as **JSON templates** in `prompts/` and loaded at runtime with optional context injection.

### Available Prompts

| File                            | Purpose                                  |
| ------------------------------- | ---------------------------------------- |
| `program_reasoning_v1.json`     | Program match reasoning narrative        |
| `scholarship_reasoning_v1.json` | Scholarship match reasoning narrative    |
| `attribution_report_v1.json`    | Structured attribution report generation |
| `research_alignment_v1.json`    | Research interest alignment analysis     |

### Prompt Utilities

The `app/utils/prompt_utils.py` module provides:

- `load_prompt_template(filename)` — load raw JSON from disk
- `merge_runtime_context(template, context)` — inject runtime data
- `build_prompt_json(filename, context)` — full pipeline, returns JSON string
- `build_prompt_text(filename, context)` — full pipeline, returns formatted text

***

## Development Workflow

### Code Quality Checks

```bash
black app/ tests/
isort app/ tests/
flake8 app/ tests/ --max-line-length=120 --extend-ignore=E203,W503,E501
pylint app/
mypy app/ --ignore-missing-imports --no-strict-optional
ALLOW_DB_FAILURE=true X_SERVICE_TOKEN=test-service-token pytest tests/ -v
```

### Pre-Commit Script

```bash
chmod +x pre-commit-check.sh
./pre-commit-check.sh
```

***

## Testing

### Run All Tests

```bash
ALLOW_DB_FAILURE=true X_SERVICE_TOKEN=test-service-token pytest tests/ -v
```

### Run with Coverage

```bash
ALLOW_DB_FAILURE=true X_SERVICE_TOKEN=test-service-token pytest tests/ --cov=app --cov-report=html -v
open htmlcov/index.html
```

### Test Structure

```
tests/
├── conftest.py                     # Shared fixtures
├── fake_repos.py                   # In-memory repository mocks
├── unit/
│   ├── test_config.py              # Configuration loading
│   ├── test_main.py                # Health endpoints
│   ├── test_security.py            # X-Service-Token validation
│   ├── test_program_scorer.py      # Program scoring logic tests
│   ├── test_scholarship_scorer.py  # Scholarship scoring logic tests
│   └── test_prompt_utils.py        # Prompt template loading & context merge
└── integration/
    ├── test_matching_attribution_flow.py  # Matching -> persistence -> attribution chain
    └── test_embedding_vector_flow.py      # Real pgvector insert/query flow
```

***

## CI/CD Pipeline

**Workflow**: `.github/workflows/deploy.yml` (GitHub Actions name: **OuroborosAI Eligibility Engine CI/CD Pipeline**)

**Triggers**:
- Pull requests to `main` or `develop`
- Pushes to `main` for image publish + post-push image scanning

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| **Format** | Black + isort validation |
| **Lint** | flake8 + pylint (both blocking) |
| **Unit Tests** | `pytest tests/unit/` with JUnit XML artifact |
| **Type Check** | mypy — blocking (after format + lint) |
| **Tests + Coverage** | Full `pytest tests/` with HTML + Cobertura XML |
| **Security Audit** | Bandit blocking scan + Snyk OSS scan artifact |
| **Docker Build** | PRs build locally; `main` pushes image to GHCR |
| **Trivy Scan** | `main` only container image scan against pushed GHCR image |
| **Summary** | Markdown table of all job results |

### Security Scan Policy

- `Bandit` runs as a blocking static security check.
- `Snyk` OSS scanning runs when `SNYK_TOKEN` is configured and uploads SARIF/artifacts.
- `Trivy` runs on `main` after the image is pushed to GHCR and scans the published container image.

---

## Deployment

### Docker Compose (Full Stack)

```bash
docker compose up --build -d
docker compose logs -f
docker compose down
docker compose down -v    # remove volumes
```

### Docker (Service Only)

```bash
docker build -t eligibility-engine .

docker run -p 8004:8004 \
  -e DB_HOST=postgres-host \
  -e DB_PASSWORD=secret \
  -e X_SERVICE_TOKEN=token \
  -e OPENAI_API_KEY=sk-... \
  eligibility-engine
```

***

## Project Structure

```
ouroboros-ai-eligibility-engine/
├── app/
│   ├── api/                         # Route handlers (thin layer)
│   │   ├── health.py                # GET / and /health
│   │   ├── matching.py              # POST /evaluate, GET /results/{user_id}
│   │   └── attribution.py           # GET /report/{match_id}
│   ├── core/                        # Infrastructure
│   │   ├── logging.py               # structlog configuration
│   │   └── security.py              # X-Service-Token validation
│   ├── llm/                         # LLM-specific logic
│   │   ├── prompts.py               # Prompt loading with context injection
│   │   ├── schemas.py               # Pydantic schemas for LLM output
│   │   ├── openai_client.py         # OpenAI client with retry + embeddings
│   │   └── anthropic_client.py      # Anthropic client with retry
│   ├── middleware/                   # Middleware
│   │   ├── service_auth.py          # X-Service-Token dependency
│   │   └── logging_middleware.py    # Request/response logging with trace_id
│   ├── models/                      # Pydantic request/response schemas
│   │   ├── common_models.py         # StandardResponse, Pagination
│   │   ├── matching_models.py       # EvaluateRequest, MatchResult, ScoreBreakdown
│   │   ├── attribution_models.py    # AttributionReport, Strength, Gap
│   │   └── embedding_models.py      # ResearchEmbedding, VectorQuery
│   ├── repositories/                # Raw SQL data access (asyncpg)
│   │   ├── db_pool.py               # PostgreSQL asyncpg connection pool
│   │   ├── postgres_base.py         # Base repository with async helpers
│   │   ├── postgres_match_repo.py   # Match results CRUD
│   │   ├── postgres_attribution_repo.py  # Attribution reports CRUD
│   │   ├── postgres_embedding_repo.py    # Vector embeddings CRUD + similarity search
│   │   └── postgres_history_repo.py      # Scoring history CRUD
│   ├── services/                    # Business logic
│   │   ├── matching_service.py      # Orchestrates program/scholarship matching
│   │   ├── scoring/                 # Scoring engines
│   │   │   ├── program_scorer.py    # Program match scoring logic
│   │   │   ├── scholarship_scorer.py  # Scholarship match scoring logic
│   │   │   └── weights.py           # Scoring weight constants
│   │   ├── explainability_service.py  # Generate attribution reports via LLM
│   │   ├── embedding_service.py     # Generate/query research embeddings
│   │   └── llm_service.py           # LLM calls for reasoning/explanation
│   ├── utils/                       # Utilities
│   │   ├── exceptions.py            # Custom exception hierarchy
│   │   ├── trace_id.py              # UUID-v4 trace ID generation
│   │   ├── helpers.py               # generate_uuid, timestamps
│   │   ├── timezone.py              # UTC helpers
│   │   └── prompt_utils.py          # JSON template loading & context merge
│   ├── config.py                    # Pydantic settings
│   └── main.py                      # FastAPI app with lifespan
├── prompts/                         # Version-controlled LLM prompt templates
│   ├── program_reasoning_v1.json
│   ├── scholarship_reasoning_v1.json
│   ├── attribution_report_v1.json
│   └── research_alignment_v1.json
├── migrations/                      # SQL migration files (001-006)
├── scripts/
│   ├── run_migrations.py
│   ├── seed_test_embeddings.py
│   └── generate_service_token.py
├── tests/                           # Test suite
├── .github/workflows/
│   └── deploy.yml                   # CI/CD pipeline
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── pytest.ini
├── .pylintrc
├── .flake8
├── .env.example
├── start.sh
├── pre-commit-check.sh
└── README.md
```

***

## Troubleshooting

### Database Connection Failed

```bash
# Check PostgreSQL is running
docker compose ps

# Test connection
psql -h localhost -p 5433 -U postgres -d ouroboros_eligibility_db -c "SELECT 1;"

# Verify credentials
grep DB_ .env
```

### pgvector Extension Missing

```bash
# Ensure pgvector is installed
psql -U postgres -d ouroboros_eligibility_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify
psql -U postgres -d ouroboros_eligibility_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### LLM Generation Failed

```bash
# Verify API keys are set
grep API_KEY .env

# Test OpenAI connectivity
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Import Errors

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

***

## Attribution

**Developed by**: OuroborosAI Developer Team

**Project**: Ouroboros AI Scholarship Discovery Platform
