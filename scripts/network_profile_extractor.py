#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

BW_RANGE      = (1.0,   50.0)
LATENCY_RANGE = (12.5,  75.0)
JITTER_RANGE  = (0.6,   3.75)
LOSS_RANGE    = (0.0,   5.0)

DEVICE_CLASSES = [
    "smart_home_hub",
    "wearable_device",
    "voice_assistant",
    "smart_appliance",
    "health_monitor",
    "smart_home_hub_2",
    "wearable_device_2",
    "voice_assistant_2",
    "smart_appliance_2",
    "health_monitor_2",
]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def scale_to_range(norm_value, lo, hi):
    return lo + norm_value * (hi - lo)


def load_features(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append([float(v) for v in row])
    return header, rows


def extract_profile(row, header):
    def col(name):
        try:
            return row[header.index(name)]
        except ValueError:
            return None

    mi_mean = col("MI_dir_0.1_mean")
    jit_mean = col("HH_jit_0.1_mean")
    hh_std   = col("HH_0.1_std_0")

    mi_mean  = mi_mean  if mi_mean  is not None else 68.0
    jit_mean = jit_mean if jit_mean is not None else 0.0
    hh_std   = hh_std   if hh_std   is not None else 0.0

    norm_bw      = clamp((mi_mean - 50.0) / 150.0, 0.0, 1.0)
    norm_latency = clamp(1.0 - norm_bw, 0.0, 1.0)
    norm_jitter  = clamp(abs(jit_mean) / 10.0, 0.0, 1.0)
    norm_loss    = clamp(hh_std / 50.0, 0.0, 1.0)

    return {
        "bandwidth_mbps": round(scale_to_range(norm_bw,      *BW_RANGE),      2),
        "latency_ms":     round(scale_to_range(norm_latency, *LATENCY_RANGE), 2),
        "jitter_ms":      round(scale_to_range(norm_jitter,  *JITTER_RANGE),  3),
        "loss_pct":       round(scale_to_range(norm_loss,    *LOSS_RANGE),    3),
    }


def netem_command(container_id, iface, profile):
    bw_kbit = int(profile["bandwidth_mbps"] * 1000)
    return (
        f"docker exec {container_id} tc qdisc replace dev {iface} root netem "
        f"delay {profile['latency_ms']}ms {profile['jitter_ms']}ms "
        f"loss {profile['loss_pct']}% rate {bw_kbit}kbit"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="datasets",
                        help="Directory containing MedBIoT CSV files")
    parser.add_argument("--iface", default="eth0",
                        help="Network interface inside containers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-devices", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    datasets_dir = Path(args.datasets)
    if not datasets_dir.exists():
        datasets_dir = Path(__file__).parent.parent / "datasets"

    csv_files = sorted(datasets_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {datasets_dir}")

    all_rows = []
    all_header = None
    for csv_file in csv_files:
        header, rows = load_features(csv_file)
        if all_header is None:
            all_header = header
        all_rows.extend(rows)

    selected = random.sample(all_rows, min(args.n_devices, len(all_rows)))

    print(f"\nPer-device tc netem commands ({args.n_devices} containers)")
    print("=" * 60)
    profiles = {}
    for i, row in enumerate(selected):
        device_class = DEVICE_CLASSES[i % len(DEVICE_CLASSES)]
        container_id = f"llceiot_iot_{i+1}"
        profile = extract_profile(row, all_header)
        profiles[container_id] = profile

        print(f"\nDevice {i+1}: {device_class}  [{container_id}]")
        print(f"  bandwidth={profile['bandwidth_mbps']} Mbps  "
              f"latency={profile['latency_ms']} ms  "
              f"jitter={profile['jitter_ms']} ms  "
              f"loss={profile['loss_pct']}%")
        print(f"  {netem_command(container_id, args.iface, profile)}")

    print(f"\nAll {args.n_devices} profile parameters are within the ranges:")
    print(f"  Bandwidth: {BW_RANGE[0]}–{BW_RANGE[1]} Mbps")
    print(f"  Latency:   {LATENCY_RANGE[0]}–{LATENCY_RANGE[1]} ms")
    print(f"  Jitter:    {JITTER_RANGE[0]}–{JITTER_RANGE[1]} ms")
    print(f"  Loss:      {LOSS_RANGE[0]}–{LOSS_RANGE[1]}%")
    print()


if __name__ == "__main__":
    main()
