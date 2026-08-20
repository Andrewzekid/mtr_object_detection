"""Config helpers for the label review tool: seed-category bootstrapping
and SAM3 device resolution. (The optional JSON config file itself is
parsed inline in ``main()``.)"""

import json
import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Category bootstrapping
# ---------------------------------------------------------------------------

def _seed_categories(seed_json: Optional[str]) -> List[Dict[str, Any]]:
    """Build the initial category list.

    If --json is given, use its categories. Otherwise start empty —
    categories are created from the side panel's Add field, and a resumed
    labels file brings its own (merged by load_existing).
    """
    cats: List[Dict[str, Any]] = []
    if seed_json and os.path.exists(seed_json):
        with open(seed_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("categories", []):
            cats.append({"id": c["id"], "name": c["name"]})
    return cats
def _resolve_device(name: str) -> str:
    """Resolve 'auto' → 'cuda' when torch reports CUDA available, else
    'cpu'. Explicit 'cuda'/'cpu' pass through unchanged."""
    if name != "auto":
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
