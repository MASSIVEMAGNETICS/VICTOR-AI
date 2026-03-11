# VICTOR-AI

Victor is a highly autonomous, locally-hosted, continuously-learning agentic AI system built under the **Ethica AI** umbrella by **Massive Magnetics**.

Victor's V1 architecture is composed of five tightly integrated subsystems — each one corresponding to a real, open-source framework deployable on your own hardware today.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          Victor Core                            │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Ego Core   │  │  Endocrine   │  │  Ethica Moral Fabric  │  │
│  │  (Memory)   │  │   System     │  │  (Guardrails)         │  │
│  │  Mem0 +     │  │  Ollama +    │  │  NeMo Guardrails +    │  │
│  │  pgvector   │  │  LiteLLM     │  │  Colang rules         │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────┐  ┌────────────────────────────┐  │
│  │  Symbiotic Bridge        │  │  Reality Translation       │  │
│  │  (Environment)           │  │  Engine (Agent)            │  │
│  │  Home Assistant +        │  │  LangGraph +               │  │
│  │  WebSockets              │  │  Custom Tools              │  │
│  └──────────────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

| # | Subsystem | Role | Technology |
|---|-----------|------|------------|
| 1 | **Ego Core** | Persistent long-term memory | Mem0 + PostgreSQL/pgvector |
| 2 | **Endocrine System** | Dynamic local compute routing | Ollama + LiteLLM |
| 3 | **Ethica Moral Fabric** | Unbreakable ethical guardrails | NVIDIA NeMo Guardrails |
| 4 | **Symbiotic Bridge** | Real-time environmental awareness | Home Assistant + WebSockets |
| 5 | **Reality Translation Engine** | Agentic task execution | LangGraph + custom tools |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- A machine with a GPU (recommended) or a capable CPU

### 1. Clone and configure

```bash
git clone https://github.com/MASSIVEMAGNETICS/VICTOR-AI.git
cd VICTOR-AI
cp .env.example .env
# Edit .env with your settings
```

### 2. Start infrastructure services

```bash
docker compose up -d
```

This starts:
- **PostgreSQL/pgvector** on port 5432 (Mem0 backing store)
- **Home Assistant** on port 8123 (environmental awareness)

### 3. Install Ollama and pull models

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3          # balanced / agent model
ollama pull phi3            # fast/lightweight tasks
ollama pull codellama       # code generation
ollama pull nomic-embed-text  # embeddings for Mem0
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Victor

```python
from victor import Victor

# Initialise with all five subsystems
v = Victor(
    user_id="your_name",
    enable_environment=True,   # Start the HA WebSocket bridge
)

# Single-turn chat (uses memory + guardrails + dynamic compute)
response = v.chat("Summarise what we talked about last week.")
print(response)

# Agentic multi-step task (uses LangGraph + tools)
result = v.run_task("Read the files in ~/Documents and write a summary email to alice@example.com")
print(result)

# Shut down background services when done
v.shutdown()
```

---

## Project Structure

```
victor/
├── __init__.py          # Package entry point; exports Victor
├── core.py              # Victor — main orchestration class
├── memory/
│   ├── __init__.py
│   └── manager.py       # MemoryManager — Mem0 + pgvector integration
├── compute/
│   ├── __init__.py
│   └── router.py        # ComputeRouter — Ollama + LiteLLM dynamic routing
├── guardrails/
│   ├── __init__.py
│   ├── ethics.py        # EthicsGuardrail — NeMo Guardrails integration
│   └── colang/
│       ├── config.yml   # NeMo Guardrails configuration
│       └── ethics.co    # Colang ethical rule definitions
├── environment/
│   ├── __init__.py
│   └── bridge.py        # EnvironmentBridge — Home Assistant WebSocket client
└── agent/
    ├── __init__.py
    ├── executor.py      # AgentExecutor — LangGraph ReAct agent
    └── tools.py         # Built-in tools: read_file, fetch_url, send_email, …

tests/                   # Pytest test suite (no live services required)
docker-compose.yml       # Infrastructure: pgvector + Home Assistant
requirements.txt         # Runtime dependencies
requirements-dev.txt     # Development/test dependencies
.env.example             # Environment variable template
```

---

## Configuration

All runtime parameters are controlled via environment variables. Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `PGVECTOR_HOST` | `localhost` | PostgreSQL host |
| `PGVECTOR_DB` | `victor` | Database name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `COMPUTE_MODEL_FAST` | `phi3` | Model for lightweight tasks |
| `COMPUTE_MODEL_BALANCED` | `llama3` | Default model |
| `COMPUTE_MODEL_HEAVY` | `llama3:70b` | Model for complex reasoning |
| `COMPUTE_MODEL_CODE` | `codellama` | Model for code generation |
| `HA_WS_URL` | `ws://localhost:8123/api/websocket` | Home Assistant WebSocket URL |
| `HA_LONG_LIVED_TOKEN` | _(empty)_ | Home Assistant long-lived access token |

---

## Running Tests

Tests use mocks and require no live services:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Ethical Guardrails

Victor's behaviour is governed by an immutable Colang rule-set (`victor/guardrails/colang/ethics.co`).  Hardcoded prohibitions include:

- Destructive shell commands (`rm -rf`, `shutdown`, `DROP DATABASE`, …)
- Requests to create malware or cause harm
- Requests involving sensitive personal/financial data
- Misrepresenting itself as human

Rules are enforced at both the **input** stage (fast keyword blocklist) and the **output** stage (full NeMo Guardrails pipeline).

---

## Agentic Tools

The LangGraph agent ships with the following built-in tools:

| Tool | Description |
|---|---|
| `read_local_file` | Read any file on the local filesystem |
| `list_directory` | List the contents of a directory |
| `fetch_url` | HTTP GET a URL and return the response body |
| `send_email` | Send a plain-text email via SMTP |
| `parse_json` | Parse and pretty-print a JSON string |

Additional tools can be added by defining new `@tool`-decorated functions in `victor/agent/tools.py` and appending them to `ALL_TOOLS`.

---

## License

Proprietary — © Massive Magnetics. All rights reserved.