#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

FAMILY_DEADLINE_RANGES = {
    "command_control": (0.5, 2.0),
    "interpretation":  (1.0, 5.0),
    "personalization": (30.0, 120.0),
}

DELTA_MAX = {
    "inference":   0.05,
    "fine_tuning": 0.10,
}


def classify_family(task):
    if task.get("task_type") == "fine_tuning":
        return "personalization"
    device = task.get("device_type", "")
    if device in ("smart_home_hub", "voice_assistant", "smart_appliance"):
        return "command_control"
    return "interpretation"


def load_tasks(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("tasks", data) if isinstance(data, dict) else data


def summarize(tasks):
    n = len(tasks)
    print(f"\nLL-CEIoT Workload Summary  ({n} tasks total)")
    print("=" * 56)

    inf_tasks = [t for t in tasks if t.get("task_type") == "inference"]
    ft_tasks  = [t for t in tasks if t.get("task_type") == "fine_tuning"]
    print(f"\nTask type split:")
    print(f"  Inference (INF):    {len(inf_tasks):3d}  ({100*len(inf_tasks)/n:.1f}%)")
    print(f"  Fine-tuning (FT):   {len(ft_tasks):3d}  ({100*len(ft_tasks)/n:.1f}%)")

    families = {"command_control": 0, "interpretation": 0, "personalization": 0}
    for t in tasks:
        families[classify_family(t)] += 1
    print(f"\nTask family breakdown:")
    print(f"  Command/control INF: {families['command_control']:3d}")
    print(f"  Interpretation INF:  {families['interpretation']:3d}")
    print(f"  Personalization FT:  {families['personalization']:3d}")

    devices = {}
    for t in tasks:
        d = t.get("device_type", "unknown")
        devices[d] = devices.get(d, 0) + 1
    print(f"\nDevice class breakdown:")
    for dev, cnt in sorted(devices.items(), key=lambda x: -x[1]):
        print(f"  {dev:<22s}: {cnt:3d}")

    deadlines = [t.get("deadline", t.get("t_dl")) for t in tasks if t.get("deadline") or t.get("t_dl")]
    if deadlines:
        print(f"\nDeadline distribution (s):")
        print(f"  Min: {min(deadlines):.1f}  Max: {max(deadlines):.1f}  "
              f"Median: {statistics.median(deadlines):.1f}  Mean: {statistics.mean(deadlines):.1f}")
        by_family = {"command_control": [], "interpretation": [], "personalization": []}
        for t in tasks:
            d = t.get("deadline", t.get("t_dl"))
            if d is not None:
                by_family[classify_family(t)].append(d)
        for fam, vals in by_family.items():
            if vals:
                lo, hi = FAMILY_DEADLINE_RANGES[fam]
                print(f"  {fam:<22s}: [{min(vals):.1f}, {max(vals):.1f}]  (paper [{lo}, {hi}])")

    prompt_lengths = [t.get("prompt_length", t.get("tau_prompt")) for t in tasks]
    gen_lengths    = [t.get("gen_length", t.get("tau_gen")) for t in tasks]
    prompt_lengths = [x for x in prompt_lengths if x is not None]
    gen_lengths    = [x for x in gen_lengths    if x is not None]
    if prompt_lengths:
        print(f"\nToken-length profiles:")
        print(f"  Prompt — min: {min(prompt_lengths)}, max: {max(prompt_lengths)}, "
              f"mean: {statistics.mean(prompt_lengths):.1f}")
    if gen_lengths:
        print(f"  Gen    — min: {min(gen_lengths)}, max: {max(gen_lengths)}, "
              f"mean: {statistics.mean(gen_lengths):.1f}")

    print(f"\nAccuracy budgets (constraints C6-C7):")
    print(f"  INF tasks: delta_max = {DELTA_MAX['inference']}")
    print(f"  FT  tasks: delta_max = {DELTA_MAX['fine_tuning']}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="configs/tasks.json")
    args = parser.parse_args()

    path = Path(args.tasks)
    if not path.exists():
        path = Path(__file__).parent.parent / "configs" / "tasks.json"

    summarize(load_tasks(path))


if __name__ == "__main__":
    main()
