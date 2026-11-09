"""Optional wall-clock budgets and outcome-blind, checkpointed resampling.

Checkpoints are trusted local pickle files, never load externally supplied ones.
They resume completed draws, not arbitrary Python process state.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


def journal(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(),
                                    monotonic=time.monotonic(), **fields), default=str) + '\n')
        stream.flush()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class BudgetExceeded(RuntimeError):
    pass


class RuntimeBudget:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / 'budget.json'
        self.events = self.root / 'timing.jsonl'

    @classmethod
    def create(cls, root: Path, initial: float = 1800, hard: float = 3600,
               tool_seconds: float = 180, request_seconds: float = 240,
               max_requests: int = 100, threads: int = 4) -> 'RuntimeBudget':
        if not (0 < initial <= hard and tool_seconds > 0 and request_seconds > 0
                and max_requests > 0 and threads > 0):
            raise ValueError('Invalid runtime limits')
        obj = cls(root)
        obj.root.mkdir(parents=True, exist_ok=True)
        if obj.path.exists():
            raise FileExistsError('Budget already exists; do not restart a paid attempt')
        now = time.time()
        atomic_json(obj.path, dict(start=now, deadline=now+initial, hard_deadline=now+hard,
                    initial_seconds=initial, hard_seconds=hard, tool_seconds=tool_seconds,
                    request_seconds=request_seconds, max_requests=max_requests, threads=threads,
                    requests=0))
        journal(obj.events, 'budget_created', initial_seconds=initial, hard_seconds=hard)
        return obj

    def state(self) -> dict:
        return json.loads(self.path.read_text(encoding='utf-8'))

    def remaining(self) -> float:
        state = self.state()
        return max(0., min(state['deadline'], state['hard_deadline']) - time.time())

    def check(self) -> None:
        if self.remaining() <= 0:
            raise BudgetExceeded('Case runtime budget exhausted')

    def feedback(self) -> str:
        s = self.state()
        return (f"RUNTIME: {self.remaining():.0f}s remaining; hard-cap remaining "
                f"{max(0,s['hard_deadline']-time.time()):.0f}s; "
                f"{s['requests']}/{s['max_requests']} model requests used. "
                f"Uncheckpointed tool limit {s['tool_seconds']}s. "
                "Use run_resamples for expensive repeated computation. Timing-based extensions "
                "are evaluated automatically after its pilot. Never extend based on significance.")

    def begin_request(self) -> tuple[str, float]:
        self.check()
        s = self.state()
        if s['requests'] >= s['max_requests']:
            raise BudgetExceeded('Model request count limit exhausted')
        s['requests'] += 1
        atomic_json(self.path, s)
        ident = uuid4().hex
        timeout = min(s['request_seconds'], self.remaining())
        journal(self.events, 'request_start', request_id=ident, timeout_seconds=timeout)
        return ident, timeout

    def reserve(self, remaining_work: float, *, evidence: dict) -> bool:
        """Only host-measured runtime evidence; never accepts p-values/results."""
        if not math.isfinite(remaining_work) or remaining_work < 0:
            raise ValueError('Invalid estimate')
        self.check()
        s = self.state()
        desired = time.time() + remaining_work * 1.5 + 120
        approved = desired <= s['hard_deadline']
        if approved and desired > s['deadline']:
            s['deadline'] = desired
            atomic_json(self.path, s)
        journal(self.events, 'budget_estimate', estimated_work_seconds=remaining_work,
                approved=approved, deadline=s['deadline'], evidence=evidence)
        return approved

    def resampler(self, session_id: str) -> Callable:
        def run_resamples(name: str, total: int, sample_fn: Callable[[int], Any], *,
                          chunk_size: int = 10, pilot_size: int = 5, seed: int = 0,
                          plan: str, resume: bool = False) -> list:
            """sample_fn(i) MUST seed local RNGs from seed+i; global RNGs are seeded here.
            Fix total and estimator in plan before seeing draws. No p-based stopping.
            Use unique names per estimator. Resume requires same function code/plan.
            """
            import marshal
            import numpy as np
            if (not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) or
                    any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
                        for x in (total, chunk_size, pilot_size)) or not plan.strip()):
                raise ValueError('Provide name, positive fixed counts and a statistical plan')
            code_hash = hashlib.sha256(marshal.dumps(sample_fn.__code__)).hexdigest()
            spec = dict(total=total, seed=seed, plan=plan, code_hash=code_hash,
                        chunk_size=chunk_size, pilot_size=pilot_size)
            folder = self.root / 'checkpoints' / session_id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (name + '.pkl')
            values: list = []
            elapsed = 0.
            if path.exists():
                if not resume:
                    raise ValueError('Checkpoint exists: use resume=True with unchanged plan/function')
                with path.open('rb') as stream:
                    saved = pickle.load(stream)
                if saved['spec'] != spec:
                    raise ValueError('Checkpoint plan/function mismatch; refusing to mix draws')
                values, elapsed = saved['values'], saved['elapsed']
            journal(self.events, 'resampling_plan', session=session_id, name=name, spec=spec,
                    completed=len(values), checkpoint=str(path))
            while len(values) < total:
                self.check()
                if values and not self.reserve(elapsed / len(values) * (total-len(values)),
                        evidence=dict(name=name, completed=len(values), total=total, measured_seconds=elapsed)):
                    raise BudgetExceeded('Measured plan exceeds hard cap. Checkpoint retained; '
                        'do not treat partial draws as a completed test or stop by significance. '
                        'Report budget infeasibility or explicitly redesign before new inference.')
                count = min(pilot_size if not values else chunk_size, total-len(values))
                journal(self.events, 'chunk_start', session=session_id, name=name,
                        completed=len(values), count=count)
                t0 = time.monotonic()
                for _ in range(count):
                    self.check()
                    index = len(values)
                    np.random.seed((seed+index) % (2**32))
                    random.seed(seed+index)
                    values.append(sample_fn(index))
                elapsed += time.monotonic()-t0
                tmp = path.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    pickle.dump(dict(spec=spec, values=values, elapsed=elapsed,
                                     numpy_rng=np.random.get_state(), python_rng=random.getstate()), stream)
                tmp.replace(path)
                journal(self.events, 'chunk_end', session=session_id, name=name,
                        completed=len(values), total=total, measured_seconds=elapsed,
                        checkpoint=str(path))
            return values
        return run_resamples


BUDGET_INSTRUCTION = """
Runtime is bounded. Expensive bootstrap/permutation/repeated cross-fitting MUST
use the supplied host helper, not a monolithic for-loop:
  def draw(i):
      rng = np.random.default_rng(123+i)  # deterministic per draw
      ... # one complete replication, preserving the chosen estimator
      return numeric_result
  draws = run_resamples('unique_name', total=200, sample_fn=draw,
      pilot_size=5, chunk_size=10, seed=123,
      plan='fixed estimator, resampling scheme, replication count and inference rule')
The host persists each completed chunk and measures the pilot to request extra
time up to the hard cap; it never bases extension on p-values. Fix total before
viewing draws. Never silently reduce total, select successful draws, or report
partial resampling as complete. Use resume=True only with the unchanged named
function and plan after an interruption; a killed Python kernel loses other
variables, so recreate definitions first. Checkpoints retain completed draws.
Chunks should take less than the uncheckpointed-tool limit; use smaller chunks
for costly fits. A single replication also has that limit. Keep diagnostics
focused; reuse identical computations. Set sklearn n_jobs=4 or less. Retain the
requested estimand and valid uncertainty calculation when proposing cheaper work.
"""
