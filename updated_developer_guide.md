# Updated Developer Guide — Sprint 3 Addendum

> This addendum supplements the existing **[`DEVELOPER-GUIDE.md`](DEVELOPER-GUIDE.md)** at the repository root. **Read the existing guide first.** Everything below describes only what changed during Sprint 3 (Full Agentic Upstream — MRO + Mobility). The base setup is unchanged.

---

## 1. What's Different in Sprint 3 (At a Glance)

| Change                                                                                                                            | Reason                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| New pinned requirements file: `requirements-agentic.txt` at the repo root                                                         | Locks the LangGraph/LangChain agentic stack so a stray `pip install` cannot break the notebooks                                |
| New shared LLM provider module: `radp/agentic/llm/provider.py`                                                                    | Single source of truth for Groq / Bedrock / OpenAI / local-Ollama selection, used by **both** Agentic MRO and Agentic Mobility |
| Three new scenario presets in `radp/digital_twin/agentic_mobility/defaults/`                                                      | Reproducible urban / suburban / rural mobility outputs                                                                         |
| New `--preset` CLI flag on `end_to_end_example.py`                                                                                | Discoverable way to run any of the three scenarios                                                                             |
| New test directories: `tests/agentic/`, `tests/agentic_mobility/`, and `apps/mobility_robustness_optimization/agentic_mro/tests/` | Validation + swap + pipeline + preset tests; all offline-safe                                                                  |

Everything else in the existing Developer Guide — Python version, virtual environment, `PYTHONPATH`, Docker, core RADP setup, services setup, MRO/CCO/Energy/LB setup — **is unchanged**. Follow it as-is.

---

## 2. Existing Steps That Are Still Required

These remain exactly as documented in `DEVELOPER-GUIDE.md`:

1. **Prerequisites** — Python 3.9–3.10, Docker, ffmpeg if you plan to use notebook video output.
2. **Virtual environment** — `python3 -m venv .venv && source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows).
3. **`PYTHONPATH`** — `export PYTHONPATH="$(pwd)":$PYTHONPATH`.
4. **Core RADP installs** — the full `pip3 install -r …` block in the Quick Start. Run them all.
5. **Docker services** — `docker compose -f dc.yml -f dc-prod.yml up -d --build` for prod, the `dc-dev.yml` overlay for dev.
6. **Notebook data** — `cd notebooks/data && unzip sim_data.zip mro_data.zip …` as before.

If you are setting up from scratch, complete those steps first, then return here.

---

## 3. New Step Added by Sprint 3 — Install the Pinned Agentic Stack

After the core RADP installs from the existing Developer Guide, run **one** additional command:

```bash
pip3 install -r requirements-agentic.txt
```

This installs the locked set of agentic-stack packages:

```
langgraph==1.2.0
langgraph-checkpoint==4.1.0
langchain-core==1.4.0
langchain-groq==1.1.2
langchain-aws==1.5.0
groq==0.37.1
boto3==1.43.11
botocore==1.43.11
pydantic==2.13.4
tenacity==9.1.4
geopy==2.4.1
```

**This file replaces the previous `pip3 install -r apps/mobility_robustness_optimization/agentic_mro/requirements.txt` line** in the Quick Start — you no longer need that older, unpinned file.

> **Why the change:** LangGraph and LangChain ship breaking releases frequently. Pinning to a known-good snapshot prevents a fresh checkout from silently grabbing an incompatible version.

### Optional: install the OpenAI or Ollama backend

The shared LLM provider supports OpenAI and a local Ollama backend, but neither package is in `requirements-agentic.txt` (so users who only need Groq or Bedrock don't pay the cost). Install them only when you need them:

```bash
# OpenAI backend
pip3 install openai

# Local model backend (Ollama via langchain-community)
pip3 install langchain-community
# And install/run Ollama from https://ollama.com
```

---

## 4. Environment Variables — What's New

The shared LLM provider (`radp/agentic/llm/provider.py`) picks its backend from either a config dict or the `LLM_PROVIDER` environment variable. Set whichever keys apply to the providers you want to use.

Add to your existing `.env` file at `radp/digital_twin/agentic_mobility/.env` (copy from `.env.example` if you haven't already):

```bash
# Pick one provider
LLM_PROVIDER=groq    # one of: groq | bedrock | openai | local

# Provider-specific keys (set only what you use)
GROQ_API_KEY=your_groq_key_here

AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_REGION=us-east-1

OPENAI_API_KEY=your_openai_key

# Local Ollama (no key needed)
OLLAMA_HOST=http://localhost:11434
```

For **Agentic MRO**, the same set of providers is selectable via `apps/mobility_robustness_optimization/agentic_mro/config.yaml`. Flip `provider:` between `groq` and `bedrock` (etc.) to swap; no code changes needed.

---

## 5. Verifying the Sprint 3 Setup Works

Run these commands in order. All five should report green.

```bash
# Activate venv and set PYTHONPATH (existing steps)
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 1. Confirm the pinned install is correct
python -c "import langgraph, langchain_core, langchain_groq, langchain_aws, groq, boto3, geopy, pydantic, tenacity; print('agentic deps OK')"

# 2. MRO validation suite (Sprint 3.1)
pytest apps/mobility_robustness_optimization/agentic_mro/tests/test_validation.py -v
# Expected: 42 passed

# 3. Shared LLM provider swap tests (Sprint 3.2)
pytest tests/agentic/test_provider_swap.py -v
# Expected: 8 passed

# 4. Mobility pipeline tests (Sprint 3.3)
pytest tests/agentic_mobility/test_pipeline.py -v
# Expected: 8 passed

# 5. Scenario preset tests (Sprint 3.4)
pytest tests/agentic_mobility/test_presets.py -v
# Expected: green
```

If any of the above fail with `ModuleNotFoundError`, your `PYTHONPATH` is not set — re-run the `export` line.

### Smoke test the new preset CLI

```bash
# Needs a GROQ_API_KEY (or Bedrock credentials) in the .env file
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset urban
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset suburban
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset rural
```

Each call should print a UE distribution summary and write a distinct CSV under `radp/digital_twin/agentic_mobility/examples/generated_ues/`.

---

## 6. Common Issues and Quick Fixes

#### 1. `ModuleNotFoundError: No module named 'radp.agentic'`

The new shared provider lives under `radp/agentic/`. If imports fail:

```bash
# Confirm PYTHONPATH includes the repo root
echo $PYTHONPATH
# If empty or missing:
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

#### 2. `ImportError: Groq SDK not installed`

You skipped `pip3 install -r requirements-agentic.txt`. Run it.

#### 3. `ImportError: Ollama support not installed`

You selected `LLM_PROVIDER=local` but did not install the optional Ollama backend.

```bash
pip3 install langchain-community
```

Then make sure the Ollama service itself is running locally (default host `http://localhost:11434`).

#### 4. `ValueError: Groq API key required` / `Unsupported LLM provider`

Either the env var is unset or the value is unknown. Open `radp/digital_twin/agentic_mobility/.env` and confirm:

- `LLM_PROVIDER` is one of `groq`, `bedrock`, `openai`, or `local`.
- The matching API key for that provider is set (Groq: `GROQ_API_KEY`, OpenAI: `OPENAI_API_KEY`, AWS: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`).

#### 5. Tests pass but emit `PydanticDeprecatedSince20` warnings

Cosmetic only — the codebase still uses Pydantic v1 idioms (`.dict()`, `.copy(update=...)`) in a few places. Safe to ignore; scheduled for the next sprint cleanup.

#### 6. Notebooks ran fine yesterday but `pip install some-package` broke them today

That's exactly the failure mode `requirements-agentic.txt` exists to prevent. Re-pin:

```bash
pip3 install -r requirements-agentic.txt --force-reinstall
```

#### 7. `--preset` flag not recognized

You're on an older copy of `end_to_end_example.py`. Pull the latest `dev` branch.

---

## 7. Quick Reference — Where the New Files Live

```
radp/agentic/                       # NEW — shared agentic infrastructure
└── llm/
    └── provider.py                 # Shared LLM provider (Groq/Bedrock/OpenAI/local)

radp/digital_twin/agentic_mobility/
└── defaults/                       # MODIFIED — three new preset files
    ├── urban.py                    # NEW
    ├── suburban.py                 # NEW
    └── rural.py                    # NEW

apps/mobility_robustness_optimization/agentic_mro/
├── llm/llm_provider.py             # NOW a re-export shim — delegates to radp.agentic.llm.provider
└── tests/
    ├── test_validation.py          # NEW — 42 MRO validation tests
    └── fixtures/                   # NEW — golden baseline CSV + JSON

tests/                              # NEW test directories
├── agentic/test_provider_swap.py
└── agentic_mobility/
    ├── test_pipeline.py
    └── test_presets.py

requirements-agentic.txt            # NEW — pinned agentic stack
```

---

## 8. Where to Read Next

- **Sprint deliverable summary:** [`sprint_implementation_report.md`](sprint_implementation_report.md) — what each task did and how to verify.
- **Agentic CLI walkthroughs:** [`README-AGENTIC.md`](README-AGENTIC.md) — unchanged, still the right place for end-to-end workflows.
- **MRO-specific usage:** [`README-MRO.md`](README-MRO.md) — unchanged.
- **Everything else (Docker, services, core RADP):** [`DEVELOPER-GUIDE.md`](DEVELOPER-GUIDE.md) — the base guide.

The Sprint 3 changes are **additive**. If you already had a working environment before this sprint, you only need to:

1. `git pull` the latest `dev` branch.
2. `pip3 install -r requirements-agentic.txt`.
3. Run the verification commands in section 5.

That's it.
