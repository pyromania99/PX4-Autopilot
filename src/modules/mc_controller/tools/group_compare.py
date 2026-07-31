#!/usr/bin/env python3
"""
Decide whether the framework differs from stock, using within-group vs
between-group spread.

The question is not "do two flights differ" - SITL flights always differ. It is
whether framework-vs-stock distances are larger than the distances stock flights
already show among themselves.

  within-stock      : how much stock varies run to run
  within-framework  : how much the framework varies run to run
  between           : framework vs stock

A signal shows a real difference only when `between` sits clearly above BOTH
within-group distributions. If within-framework is comparable to within-stock and
between overlaps them, the signal is noise-dominated.

Usage: group_compare.py --stock a.ulg b.ulg ... --fw x.ulg y.ulg ...
"""
import itertools
import sys

import numpy as np
from pyulog import ULog

AIR_ALT = 4.0

SIGNALS = [
    ("vehicle_torque_setpoint", "xyz[0]", "torque roll"),
    ("vehicle_torque_setpoint", "xyz[1]", "torque pitch"),
    ("vehicle_torque_setpoint", "xyz[2]", "torque yaw"),
    ("vehicle_thrust_setpoint", "xyz[2]", "thrust z"),
    ("vehicle_rates_setpoint", "roll", "rate sp roll"),
    ("vehicle_rates_setpoint", "pitch", "rate sp pitch"),
    ("vehicle_rates_setpoint", "yaw", "rate sp yaw"),
]


def get(u, topic):
    d = next((x for x in u.data_list if x.name == topic), None)
    return d.data if d else None


def air_window(u):
    lp = get(u, "vehicle_local_position")
    t = np.asarray(lp["timestamp"], dtype=float) / 1e6
    z = np.asarray(lp["z"], dtype=float)
    above = (-z) >= AIR_ALT
    if not np.any(above):
        return None
    idx = np.where(above)[0]
    return t[idx[0]], t[idx[-1]]


def rms(u, topic, field, w):
    d = get(u, topic)
    if d is None or field not in d:
        return None
    t = np.asarray(d["timestamp"], dtype=float) / 1e6
    v = np.asarray(d[field], dtype=float)
    m = (t >= w[0]) & (t <= w[1]) & np.isfinite(v)
    if not np.any(m):
        return None
    return float(np.sqrt(np.mean(v[m] ** 2)))


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-4) * 100


def main():
    args = sys.argv[1:]
    si, fi = args.index("--stock"), args.index("--fw")
    stock_paths = args[si + 1:fi]
    fw_paths = args[fi + 1:]

    us = [ULog(p) for p in stock_paths]
    uf = [ULog(p) for p in fw_paths]
    wins = [air_window(u) for u in us + uf]
    dur = min(w[1] - w[0] for w in wins if w)
    ws = [(w[0], w[0] + dur) for w in wins[:len(us)]]
    wf = [(w[0], w[0] + dur) for w in wins[len(us):]]

    print("=" * 90)
    print(f"{len(us)} stock flights, {len(uf)} framework flights, {dur:.0f}s aligned window")
    print("=" * 90)
    print(f"{'signal':<16}{'within-stock':>22}{'within-fw':>20}{'between':>22}  verdict")
    print(f"{'':<16}{'max (mean)':>22}{'max (mean)':>20}{'max (mean)':>22}")
    print("-" * 90)

    verdicts = {}
    for topic, field, label in SIGNALS:
        rs = [rms(u, topic, field, w) for u, w in zip(us, ws)]
        rf = [rms(u, topic, field, w) for u, w in zip(uf, wf)]
        if any(r is None for r in rs + rf):
            print(f"{label:<16}UNAVAILABLE")
            continue

        wstock = [rel(a, b) for a, b in itertools.combinations(rs, 2)]
        wfw = [rel(a, b) for a, b in itertools.combinations(rf, 2)] or [0.0]
        between = [rel(a, b) for a in rf for b in rs]

        ws_max, ws_mean = max(wstock), sum(wstock) / len(wstock)
        wf_max, wf_mean = max(wfw), sum(wfw) / len(wfw)
        bt_max, bt_mean = max(between), sum(between) / len(between)

        # Real difference only if between clearly exceeds both within-group maxima.
        within_max = max(ws_max, wf_max)
        real = bt_mean > 1.5 * max(ws_mean, wf_mean) and bt_max > within_max
        verdicts[label] = real

        print(f"{label:<16}{ws_max:>14.1f} ({ws_mean:>4.1f}){wf_max:>12.1f} ({wf_mean:>4.1f})"
              f"{bt_max:>14.1f} ({bt_mean:>4.1f})  "
              f"{'REAL DIFFERENCE' if real else 'noise-dominated'}")

    print("=" * 90)
    real_signals = [k for k, v in verdicts.items() if v]
    if real_signals:
        print("SIGNALS WITH A DIFFERENCE BEYOND RUN-TO-RUN NOISE:", ", ".join(real_signals))
        return 1
    print("NO SIGNAL DIFFERS BEYOND RUN-TO-RUN NOISE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
