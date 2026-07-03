#    core/hyperparameter_search.py - Hyperparameter search for YOLO training.
#
#    USAGE:
#        from core.hyperparameter_search import (
#            HyperparameterSearcher, SearchSpace, TrialResult,
#            GridSearchStrategy, RandomSearchStrategy, OPTIMIZE_HIGHER,
#        )
#
#        space = SearchSpace()
#        space.add("lr0", "float", low=1e-4, high=1e-1, log=True)
#        space.add("batch_size", "categorical", choices=[8, 16, 32])
#        space.add("optimizer", "categorical", choices=["SGD", "Adam", "AdamW"])
#
#        fixed = {"epochs": 50, "imgsz": 640, "model_type": "yolov8n"}
#
#        def train_fn(params, trial_dir, progress_cb, status_cb, log_cb, is_cancelled):
#            ... call ModelTrainer.train_yolo(...) ...
#            return {"success": True, "metrics": {"map50_95": 0.42}}
#
#        searcher = HyperparameterSearcher(
#            search_space=space,
#            train_fn=train_fn,
#            fixed_params=fixed,
#            strategy="random",   # "grid" or "random"
#            n_trials=20,
#            output_dir=Path("output/training/hp_search"),
#            metric="map50_95",
#            mode="max",
#        )
#        result = searcher.run(progress_callback=..., status_callback=..., ...)
#
#        print(result["best_params"], result["best_score"])
#
#    REQUIREMENTS:
#        numpy (for random sampling)
#        PyYAML (for persistence)
#
#    STRATEGIES:
#        - "grid":    exhaustively iterate every combination of discrete choices.
#        - "random":  sample `n_trials` configurations from the search space.

"""
Hyperparameter search utilities for YOLO model training.

Provides:
  * SearchSpace — declarative description of the search space.
  * GridSearchStrategy, RandomSearchStrategy — concrete strategies.
  * TrialResult — per-trial outcome container.
  * HyperparameterSearcher — orchestrator that runs trials and tracks the best.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# Convenience constants
OPTIMIZE_HIGHER = "max"
OPTIMIZE_LOWER = "min"


# =============================================================================
# Search space definition
# =============================================================================

VALID_PARAM_TYPES = {"int", "float", "categorical"}


class SearchSpace:
    """Declarative description of a hyperparameter search space.

    Each parameter is a dict with:
      - type: "int" | "float" | "categorical"
      - low / high (int/float)  OR  choices (categorical)
      - step (optional, int/float): grid step size (default: 1 for int, None for float)
      - log (bool, optional): if True, sample log-uniform between low and high
      - default (optional): default value to use if the parameter is not tuned
    """

    def __init__(self):
        self._params: List[Dict[str, Any]] = []

    def add(
        self,
        name: str,
        param_type: str,
        *,
        low: Optional[Union[int, float]] = None,
        high: Optional[Union[int, float]] = None,
        step: Optional[Union[int, float]] = None,
        log: bool = False,
        choices: Optional[Sequence[Any]] = None,
        default: Any = None,
    ) -> "SearchSpace":
        """Add a parameter to the search space."""
        if param_type not in VALID_PARAM_TYPES:
            raise ValueError(
                f"Invalid param_type '{param_type}' for '{name}'. "
                f"Must be one of {VALID_PARAM_TYPES}."
            )

        if param_type in ("int", "float"):
            if low is None or high is None:
                raise ValueError(
                    f"Numeric parameter '{name}' requires both 'low' and 'high'."
                )
            if high < low:
                raise ValueError(
                    f"Parameter '{name}': 'high' must be >= 'low'."
                )
        elif param_type == "categorical":
            if not choices:
                raise ValueError(
                    f"Categorical parameter '{name}' requires non-empty 'choices'."
                )

        spec: Dict[str, Any] = {
            "name": name,
            "type": param_type,
            "low": low,
            "high": high,
            "step": step,
            "log": log,
            "choices": list(choices) if choices is not None else None,
            "default": default,
        }
        self._params.append(spec)
        return self

    @property
    def names(self) -> List[str]:
        return [p["name"] for p in self._params]

    def __len__(self) -> int:
        return len(self._params)

    def __iter__(self):
        return iter(self._params)

    def __contains__(self, name: str) -> bool:
        return any(p["name"] == name for p in self._params)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        for p in self._params:
            if p["name"] == name:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"params": list(self._params)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSpace":
        space = cls()
        for spec in data.get("params", []):
            space.add(
                name=spec["name"],
                param_type=spec["type"],
                low=spec.get("low"),
                high=spec.get("high"),
                step=spec.get("step"),
                log=spec.get("log", False),
                choices=spec.get("choices"),
                default=spec.get("default"),
            )
        return space

    def defaults(self) -> Dict[str, Any]:
        """Return defaults for all parameters (tuned or not)."""
        return {p["name"]: p.get("default") for p in self._params}


# =============================================================================
# Trial result
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""

    trial_id: int
    params: Dict[str, Any]
    score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0
    success: bool = False
    error: Optional[str] = None
    status: str = "pending"  # pending | running | success | failed | cancelled
    model_path: Optional[str] = None
    trial_dir: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Search strategies
# =============================================================================

class BaseSearchStrategy:
    """Base class for search strategies."""

    def __init__(self, search_space: SearchSpace, seed: Optional[int] = None):
        self.search_space = search_space
        self.rng = random.Random(seed)

    def sample(self) -> Dict[str, Any]:
        """Sample a single configuration from the search space."""
        raise NotImplementedError

    def total_trials(self) -> Optional[int]:
        """Total number of trials the strategy will run, if known up-front."""
        raise NotImplementedError


class GridSearchStrategy(BaseSearchStrategy):
    """Exhaustively enumerate every combination of values in the search space.

    Continuous parameters are discretised using their `step` (int) or
    `n_steps` (float) field; if no step is provided we default to 5 evenly
    spaced points in the [low, high] interval.
    """

    DEFAULT_FLOAT_GRID_POINTS = 5

    def __init__(
        self,
        search_space: SearchSpace,
        n_steps: int = DEFAULT_FLOAT_GRID_POINTS,
        seed: Optional[int] = None,
    ):
        super().__init__(search_space, seed=seed)
        self.n_steps = max(2, n_steps)
        self._grid = self._build_grid()

    def _build_grid(self) -> List[Dict[str, Any]]:
        per_param_values: List[List[Any]] = []
        for spec in self.search_space:
            per_param_values.append(self._values_for(spec))
        combos: List[Dict[str, Any]] = []
        for combo in itertools.product(*per_param_values):
            combos.append({spec["name"]: v for spec, v in zip(self.search_space, combo)})
        return combos

    def _values_for(self, spec: Dict[str, Any]) -> List[Any]:
        ptype = spec["type"]
        if ptype == "categorical":
            return list(spec["choices"])
        if ptype == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec["step"]) if spec.get("step") is not None else 1
            step = max(1, step)
            # Make sure we always include the upper bound.
            values = list(range(low, high + 1, step))
            if values[-1] != high:
                values.append(high)
            return values
        # float
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        if step is not None and step > 0:
            values: List[float] = []
            v = low
            # Avoid infinite loops due to floating point.
            for _ in range(10_000):
                values.append(round(v, 8))
                v += float(step)
                if v > high:
                    break
            if values[-1] < high:
                values.append(round(high, 8))
            return values
        # No step -> sample n_steps evenly (optionally log-spaced).
        n = self.n_steps
        if spec.get("log"):
            if low <= 0 or high <= 0:
                raise ValueError(
                    f"Log-spaced parameter '{spec['name']}' needs positive low/high."
                )
            log_low, log_high = math.log(low), math.log(high)
            return [math.exp(log_low + (log_high - log_low) * i / (n - 1)) for i in range(n)]
        step_f = (high - low) / (n - 1) if n > 1 else 0
        return [round(low + step_f * i, 8) for i in range(n)]

    def sample(self) -> Dict[str, Any]:
        if not self._grid:
            return {}
        return dict(self.rng.choice(self._grid))

    def total_trials(self) -> int:
        return len(self._grid)


class RandomSearchStrategy(BaseSearchStrategy):
    """Randomly sample N configurations from the search space.

    Supports numeric (uniform / log-uniform) and categorical parameters.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__(search_space, seed=seed)
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        self.n_trials = int(n_trials)

    def sample(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for spec in self.search_space:
            config[spec["name"]] = self._sample_one(spec)
        return config

    def _sample_one(self, spec: Dict[str, Any]) -> Any:
        ptype = spec["type"]
        if ptype == "categorical":
            return self.rng.choice(list(spec["choices"]))
        if ptype == "int":
            return self.rng.randint(int(spec["low"]), int(spec["high"]))
        # float
        low = float(spec["low"])
        high = float(spec["high"])
        if spec.get("log"):
            if low <= 0 or high <= 0:
                raise ValueError(
                    f"Log-uniform parameter '{spec['name']}' needs positive low/high."
                )
            log_low, log_high = math.log(low), math.log(high)
            return math.exp(self.rng.uniform(log_low, log_high))
        return self.rng.uniform(low, high)

    def total_trials(self) -> int:
        return self.n_trials


def build_strategy(
    name: str,
    search_space: SearchSpace,
    *,
    n_trials: int = 20,
    n_steps: int = GridSearchStrategy.DEFAULT_FLOAT_GRID_POINTS,
    seed: Optional[int] = None,
) -> BaseSearchStrategy:
    """Factory for the supported strategies."""
    if name == "grid":
        return GridSearchStrategy(search_space, n_steps=n_steps, seed=seed)
    if name == "random":
        return RandomSearchStrategy(search_space, n_trials=n_trials, seed=seed)
    raise ValueError(f"Unknown search strategy: {name!r}. Use 'grid' or 'random'.")


# =============================================================================
# Searcher
# =============================================================================

TrainFn = Callable[
    [
        Dict[str, Any],     # tuned + fixed params
        Path,               # trial output directory
        Callable[[int], None],   # progress_callback
        Callable[[str], None],   # status_callback
        Callable[[str], None],   # log_callback
        Callable[[], bool],     # is_cancelled
    ],
    Dict[str, Any],
]


class HyperparameterSearcher:
    """Run a hyperparameter search by repeatedly calling a train function.

    The user supplies:
      * `search_space` — which hyperparameters to tune
      * `fixed_params` — values for the non-tuned hyperparameters
      * `train_fn`     — function that runs a single training run and
                         returns a dict containing at least `{"success": bool,
                         "metrics": {...}, ...}`
      * `metric`       — name of the metric to optimise in the returned
                         `metrics` dict
      * `mode`         — "max" (default) or "min"
      * `strategy`     — "grid" or "random"
      * `n_trials`     — number of trials (random) or grid resolution (grid)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        train_fn: TrainFn,
        fixed_params: Optional[Dict[str, Any]] = None,
        *,
        strategy: str = "random",
        n_trials: int = 20,
        n_steps: int = GridSearchStrategy.DEFAULT_FLOAT_GRID_POINTS,
        metric: str = "map50_95",
        mode: str = OPTIMIZE_HIGHER,
        output_dir: Optional[Union[str, Path]] = None,
        name: str = "hp_search",
        seed: Optional[int] = None,
    ):
        if mode not in (OPTIMIZE_HIGHER, OPTIMIZE_LOWER):
            raise ValueError(f"Invalid mode '{mode}'. Use 'max' or 'min'.")

        self.search_space = search_space
        self.fixed_params = dict(fixed_params or {})
        self.train_fn = train_fn
        self.metric = metric
        self.mode = mode
        self.output_dir = Path(output_dir) if output_dir else Path("output/training/hp_search")
        self.name = name
        self.seed = seed

        self.strategy = build_strategy(
            strategy,
            search_space,
            n_trials=n_trials,
            n_steps=n_steps,
            seed=seed,
        )

        self.trials: List[TrialResult] = []
        self.best: Optional[TrialResult] = None

    @property
    def total_trials(self) -> int:
        return self.strategy.total_trials() or 0

    # ----- scoring helpers -----
    def _is_better(self, new: float, current: Optional[float]) -> bool:
        if current is None:
            return True
        if self.mode == OPTIMIZE_HIGHER:
            return new > current
        return new < current

    def _coerce(self, spec: Dict[str, Any], value: Any) -> Any:
        """Cast a sampled value to the parameter's declared type."""
        ptype = spec["type"]
        if ptype == "int":
            return int(round(float(value)))
        if ptype == "float":
            return float(value)
        return value  # categorical — already typed

    def _merge(self, sampled: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(self.fixed_params)
        for spec in self.search_space:
            if spec["name"] in sampled:
                merged[spec["name"]] = self._coerce(spec, sampled[spec["name"]])
        return merged

    # ----- persistence -----
    def _write_summary(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "name": self.name,
            "metric": self.metric,
            "mode": self.mode,
            "strategy": type(self.strategy).__name__,
            "total_trials": self.total_trials,
            "completed_trials": sum(1 for t in self.trials if t.status in ("success", "failed", "cancelled")),
            "best": self.best.to_dict() if self.best else None,
            "trials": [t.to_dict() for t in self.trials],
            "search_space": self.search_space.to_dict(),
            "fixed_params": self.fixed_params,
            "updated_at": time.time(),
        }
        path = self.output_dir / "search_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # ----- main entry point -----
    def run(
        self,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run the search. Returns a result dictionary with all trials and the best."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trials = []
        self.best = None

        total = self.total_trials
        if total <= 0:
            msg = "Search space is empty — nothing to do."
            if log_callback:
                log_callback(msg)
            if status_callback:
                status_callback(msg)
            return {
                "success": True,
                "best_params": {},
                "best_score": None,
                "best_metrics": {},
                "trials": [],
                "total_trials": 0,
            }

        if status_callback:
            status_callback(
                f"Starting {type(self.strategy).__name__} search "
                f"({total} trial{'s' if total != 1 else ''}, "
                f"optimising {self.metric!r}={'max' if self.mode == 'max' else 'min'})"
            )
        if log_callback:
            log_callback(
                f"Search space: {[p['name'] for p in self.search_space]} | "
                f"Fixed: {sorted(self.fixed_params.keys())}"
            )

        for trial_id in range(total):
            if is_cancelled and is_cancelled():
                if status_callback:
                    status_callback("Search cancelled")
                break

            sampled = self.strategy.sample()
            params = self._merge(sampled)

            trial_dir = self.output_dir / f"trial_{trial_id:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            trial = TrialResult(
                trial_id=trial_id,
                params=params,
                trial_dir=str(trial_dir),
                started_at=time.time(),
                status="running",
            )
            self.trials.append(trial)

            if status_callback:
                status_callback(
                    f"Trial {trial_id + 1}/{total}: {self._format_params(params)}"
                )
            if log_callback:
                log_callback(
                    f"--- Trial {trial_id + 1}/{total} ---\n"
                    f"  Tuned params: {sampled}\n"
                    f"  Output dir:   {trial_dir}"
                )

            # Per-trial progress (0..100) wraps into the overall search progress.
            def _trial_progress(p: int, _tid: int = trial_id) -> None:
                if progress_callback:
                    # 0..100% within this trial, scaled to overall range.
                    overall = int((_tid + p / 100.0) / total * 100)
                    progress_callback(max(0, min(100, overall)))

            try:
                result = self.train_fn(
                    params,
                    trial_dir,
                    _trial_progress,
                    lambda msg, _tid=trial_id: (
                        log_callback(f"[trial {_tid + 1}] {msg}") if log_callback else None
                    ),
                    lambda msg, _tid=trial_id: (
                        log_callback(f"[trial {_tid + 1}] {msg}") if log_callback else None
                    ),
                    is_cancelled or (lambda: False),
                )
            except Exception as exc:  # pragma: no cover - defensive
                result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}

            trial.finished_at = time.time()
            trial.duration_sec = trial.finished_at - (trial.started_at or trial.finished_at)
            trial.model_path = result.get("model_path") if isinstance(result, dict) else None
            trial.metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            trial.success = bool(result.get("success")) if isinstance(result, dict) else False
            trial.error = result.get("error") if isinstance(result, dict) else "Unknown error"

            # Read score
            raw_score = trial.metrics.get(self.metric) if trial.metrics else None
            try:
                trial.score = float(raw_score) if raw_score is not None else None
            except (TypeError, ValueError):
                trial.score = None

            if trial.success:
                trial.status = "success"
                if trial.score is not None and self._is_better(trial.score, self.best.score if self.best else None):
                    self.best = trial
                    if status_callback:
                        status_callback(
                            f"  → New best {self.metric}={trial.score:.4f} "
                            f"@ trial {trial_id + 1}"
                        )
                if log_callback:
                    log_callback(
                        f"  → score={trial.score!r}, metrics={trial.metrics}"
                    )
            else:
                trial.status = "failed"
                if log_callback:
                    log_callback(f"  → FAILED: {trial.error}")

            # Persist per-trial result
            with open(trial_dir / "result.json", "w") as f:
                json.dump(trial.to_dict(), f, indent=2, default=str)
            self._write_summary()

            if progress_callback:
                progress_callback(int(((trial_id + 1) / total) * 100))

        if self.best is not None:
            best_summary = {
                "params": self.best.params,
                "score": self.best.score,
                "metrics": self.best.metrics,
                "trial_id": self.best.trial_id,
            }
        else:
            best_summary = {"params": {}, "score": None, "metrics": {}, "trial_id": None}

        result = {
            "success": any(t.status == "success" for t in self.trials),
            "best_params": best_summary["params"],
            "best_score": best_summary["score"],
            "best_metrics": best_summary["metrics"],
            "best_trial_id": best_summary["trial_id"],
            "trials": [t.to_dict() for t in self.trials],
            "total_trials": total,
            "completed_trials": sum(1 for t in self.trials if t.status != "running"),
            "metric": self.metric,
            "mode": self.mode,
            "strategy": type(self.strategy).__name__,
        }
        self._write_summary()
        if status_callback:
            if self.best is not None:
                status_callback(
                    f"Search complete. Best {self.metric}={self.best.score:.4f} "
                    f"(trial {self.best.trial_id + 1})"
                )
            else:
                status_callback("Search complete. No successful trial.")
        return result

    # ----- helpers -----
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Compact, human-readable rendering of a parameter dict."""
        parts: List[str] = []
        for k, v in params.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4g}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)


# =============================================================================
# Convenience: build a YOLO training function
# =============================================================================

def make_yolo_train_fn(
    config_path: str,
    *,
    base_kwargs: Optional[Dict[str, Any]] = None,
):
    """Return a `train_fn` that invokes `ModelTrainer.train_yolo` per trial.

    The returned function honours the standard callback protocol used by
    `HyperparameterSearcher` and returns the metrics dict that the trainer
    produces.  `base_kwargs` provides the parameters that the user is NOT
    tuning (e.g. epochs, imgsz, model_type, device, etc.).
    """
    from core.model_trainer import ModelTrainer  # local import to avoid cycles

    base = dict(base_kwargs or {})

    def train_fn(params, trial_dir, progress_cb, status_cb, log_cb, is_cancelled):
        kwargs = dict(base)
        kwargs.update(params)
        # Each trial writes into its own directory under the search run dir.
        kwargs["output_dir"] = trial_dir
        trainer = ModelTrainer(config_path=config_path, output_dir=trial_dir)
        try:
            return trainer.train_yolo(
                progress_callback=progress_cb,
                status_callback=status_cb,
                log_callback=log_cb,
                is_cancelled=is_cancelled,
                **kwargs,
            )
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    return train_fn
