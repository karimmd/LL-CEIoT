#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from typing import List

DELTA_MAX = {
    "inference":   0.05,
    "fine_tuning": 0.10,
}


@dataclass
class TaskResult:
    task_id: str
    task_type: str
    t_completion: float
    t_deadline: float
    delta_acc: float


def success(result: TaskResult) -> int:
    limit = DELTA_MAX[result.task_type]
    return int(result.t_completion <= result.t_deadline and result.delta_acc <= limit)


def compute_metrics(results: List[TaskResult]) -> dict:
    n = len(results)
    success_flags  = np.array([success(r) for r in results])
    deadline_miss  = np.array([int(r.t_completion > r.t_deadline) for r in results])
    acc_fail       = np.array([int(r.delta_acc > DELTA_MAX[r.task_type]) for r in results])

    sr    = success_flags.mean()
    er    = 1.0 - sr
    er_dl = deadline_miss.mean()
    er_ac = acc_fail.mean()

    return {
        "n_tasks":       n,
        "SR":            float(sr),
        "ER":            float(er),
        "ER_deadline":   float(er_dl),
        "ER_accuracy":   float(er_ac),
        "ER_disjoint":   bool(np.logical_and(deadline_miss, acc_fail).sum() == 0),
    }


def decompose_by_type(results: List[TaskResult]) -> dict:
    out = {}
    for ttype in ("inference", "fine_tuning"):
        subset = [r for r in results if r.task_type == ttype]
        if subset:
            out[ttype] = compute_metrics(subset)
    return out


def print_report(results: List[TaskResult]) -> None:
    agg = compute_metrics(results)
    print(f"\nEvaluation Metrics  (n={agg['n_tasks']})")
    print("=" * 40)
    print(f"  SR            : {agg['SR']:.4f}  ({100*agg['SR']:.2f}%)")
    print(f"  ER            : {agg['ER']:.4f}")
    print(f"  ER_deadline   : {agg['ER_deadline']:.4f}")
    print(f"  ER_accuracy   : {agg['ER_accuracy']:.4f}")
    print(f"  Disjoint modes: {agg['ER_disjoint']}")

    per_type = decompose_by_type(results)
    if per_type:
        print("\nPer task type:")
        for ttype, m in per_type.items():
            print(f"  {ttype:<12s}  SR={m['SR']:.3f}  "
                  f"ER_dl={m['ER_deadline']:.3f}  ER_acc={m['ER_accuracy']:.3f}")
    print()
