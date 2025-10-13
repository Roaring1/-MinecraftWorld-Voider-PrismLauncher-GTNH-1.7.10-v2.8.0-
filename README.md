# GTNH World Voider (1.7.x)

**What it does**

- Safely voids chunks with provider-safe biome handling:
  - **Overworld:** keep `Level.Biomes` if present; otherwise write Plains (id=1).
  - **Non-overworld:** preserve `Level.Biomes` if present; otherwise omit the tag to let the provider recompute.
- Chunk selection UI (`chunkmap_ui.py`) for include/exclude by dimension.
- Rules Editor with optional NEI Dump import.
- “Reset Biomes” maintenance (Tools menu).

**Run (dev)**

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main_voider.py
