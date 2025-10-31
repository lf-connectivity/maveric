## Agentic Mobility & MRO CLI Guide

> Before running any scripts or notebooks, complete the setup steps in `DEVELOPER-GUIDE.md` at the repository root. That guide covers system prerequisites, Python environment creation, dependency installation order, Docker services, and `PYTHONPATH` configuration. The instructions below assume that baseline setup is finished and you are working from an activated virtual environment in the project root.

---

### Prerequisites Recap (from the Developer Guide)

- Python 3.9–3.10 in a dedicated virtual environment (`python3 -m venv .venv && source .venv/bin/activate`).
- Repository root added to `PYTHONPATH`, e.g. `export PYTHONPATH="$(pwd)":$PYTHONPATH` on macOS/Linux.
- Core dependencies installed via the Developer Guide sequence, especially:
  - `pip install -r radp/digital_twin/requirements.txt`
  - `pip install -r notebooks/requirements.txt`
  - `pip install -r apps/mobility_robustness_optimization/agentic_mro/requirements.txt`
- Optional: GPU-enabled PyTorch (see Developer Guide) and Docker-based RADP services if you plan to integrate with live simulations.

#### API Keys & Environment Files

- Agentic mobility flows expect a `.env` file in `radp/digital_twin/agentic_mobility/`:
  ```bash
  cp radp/digital_twin/agentic_mobility/.env.example radp/digital_twin/agentic_mobility/.env
  # Edit the file and add: GROQ_API_KEY=your_key_here
  ```
- Agentic MRO reads the provider key from `apps/mobility_robustness_optimization/agentic_mro/config.yaml` (or from CLI flags). Replace the placeholder `PUT_YOUR_GROQ_API_KEY_HERE` with a valid key before running.

---

### Agentic Mobility CLI Workflow

This pipeline converts natural-language mobility requests into RADP-compatible simulations.

1. **Activate your environment and set the project root**
   ```bash
   cd /Users/tanzimfarhan/Desktop/Maveric/AgenticMaveric
   source .venv/bin/activate  # or the equivalent for your OS
   export PYTHONPATH="$(pwd)":$PYTHONPATH
   ```

2. **Generate mobility parameters only** (quick validation):
   ```bash
   python radp/digital_twin/agentic_mobility/examples/gen_mobility_example.py
   ```
   - Uses `radp/digital_twin/agentic_mobility/api.py` to turn queries into structured parameters.
   - Console output shows status, retry count, and the generated parameter JSON snippet.

3. **Run the end-to-end simulator** (natural language → CSV/JSON outputs):
   ```bash
   python radp/digital_twin/agentic_mobility/examples/end_to_end_example.py
   ```
   - Produces UE mobility tracks under `radp/digital_twin/agentic_mobility/examples/generated_ues/`.
   - Outputs include `*.csv` mobility traces and `*_metadata.json` files containing query intent, spatial bounds, and distribution breakdowns.

4. **Full pipeline with topology & visualization (optional)**:
   ```bash
   python radp/digital_twin/agentic_mobility/examples/end_to_end_viz_example.py
   ```
   - Invokes `TopologyGenerator` to place cells and emits plots/HTML dashboards (requires additional visualization dependencies from `notebooks/requirements.txt`).

5. **Feeding MRO**: Copy or symlink the generated mobility CSV and `cell_topology.csv` into `data/agentic_data/mro/` (the paths referenced by the Agentic MRO configuration) or update `apps/mobility_robustness_optimization/agentic_mro/config.yaml` to point at your files.

Key modules involved:
- `radp/digital_twin/agentic_mobility/integration.py`: orchestrates natural language parsing, validation, and UE track generation.
- `radp/digital_twin/agentic_mobility/topology_generator.py`: synthesizes tower placements from mobility metadata.

---

### Agentic MRO CLI Workflow

The Agentic MRO pipeline optimizes hysteresis and time-to-trigger parameters using a LangGraph multi-agent workflow.

#### Option A: Configuration-driven run

```bash
python apps/mobility_robustness_optimization/agentic_mro/run_from_config.py
```

What it does:
- Loads provider and optimization settings from `config.yaml` in the same directory.
- Resolves data paths relative to the repository root (defaults to `data/agentic_data/mro/ue_mobility_data.csv` and `cell_topology.csv`).
- Preprocesses the UE mobility and topology data into `apps/mobility_robustness_optimization/agentic_mro/data/sim_data.csv` using `notebooks.radp_library.preprocess_ue_data()`.
- Runs `run_agentic_mro()` from `apps/mobility_robustness_optimization/agentic_mro/main.py` and reports the best hysteresis, TTT, and score.
- Generates static plots in `apps/mobility_robustness_optimization/agentic_mro/plots/` using `utils/visuals.py` helpers (scatter map plus per-UE SINR plots).

Before launching:
- Ensure `config.yaml` has a real API key under the selected provider.
- Confirm the mobility/topology CSV paths exist (use outputs from the mobility pipeline or the provided sample data).

#### Option B: Direct CLI invocation

```bash
python apps/mobility_robustness_optimization/agentic_mro/main.py \
  --csv data/agentic_data/mro/sim_data.csv \
  --provider groq \
  --model openai/gpt-oss-120b \
  --api-key "$GROQ_API_KEY" \
  --target-score 48 \
  --max-iterations 10 \
  --rlf-threshold -4.0
```

- Pass a preprocessed simulation CSV (you can reuse the file produced by `run_from_config.py` or preprocess manually).
- Override provider/model/token parameters as needed; unsupported options fall back to the defaults defined in `main.py`.
- Use `--output path/to/results.json` to save the optimization summary.

Supporting components to know:
- `graph.py`: builds the LangGraph workflow connecting the Analyzer, Strategy, Coordinator, and Finalize nodes (`nodes/*.py`).
- `state.py`: defines the shared state schema and helpers for iterations, score tracking, and stop conditions (`utils/stop_conditions.py`).
- `utils/evaluation.py` and `utils/feature_extraction.py`: compute KPIs and summarize candidate parameter sets for the agents.

---

### Notebook Reference

Both notebooks assume you already completed the prerequisites (virtual environment, dependencies, API keys, and `PYTHONPATH`). Launch with `jupyter notebook` from the project root.

- `notebooks/agentic_mobility_model.ipynb`
  - Showcases the entire natural language → mobility → topology → visualization workflow.
  - Sections mirror the CLI scripts but provide richer previews: Matplotlib tracks, scenario comparisons, location validation, world map export, and Plotly dashboards.
  - Generates artifacts under `data/agentic_data/mobility/` (CSV/JSON, PNGs, HTML dashboards).

- `notebooks/agentic_mro.ipynb`
  - Walks through the combined pipeline: mobility generation, topology creation, preprocessing, agentic MRO optimization, and visualization.
  - Writes mobility/topology outputs to `data/agentic_data/mro/`, preprocesses to `sim_data.csv`, and saves scatter plus SINR plots under `data/agentic_data/mro/plots/`.
  - Useful for exploring intermediate DataFrames (`sim_data`, metadata), tuning optimization parameters interactively, and inspecting Plotly outputs.

Tips for both notebooks:
- Keep API rate limits in mind; LLM and geocoding calls can take several seconds.
- Ensure the virtual environment kernel is selected (`Kernel > Change Kernel > .venv` or equivalent).
- If you adjust file paths, update `config.yaml` or CLI flags to keep the notebooks and scripts aligned.

---

### Next Steps & Verification

- After running either pipeline, inspect generated CSV/JSON files and plots to confirm expected UE distributions and optimized parameters.
- Version-control any changes to `config.yaml` or derived datasets as needed for reproducibility.
- Consider wrapping CLI calls in Makefile targets or shell scripts for repeatable experimentation (after verifying via the Developer Guide’s testing recommendations).

For questions about core RADP services, additional applications (Energy Savings, Load Balancing, CCO), or extended setup options, return to `DEVELOPER-GUIDE.md` and the application-specific READMEs.

