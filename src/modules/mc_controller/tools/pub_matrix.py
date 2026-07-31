#!/usr/bin/env python3
"""Publication matrix gate.

For each control-level segment of a flight, count samples of each intermediate
topic. The framework must publish the same topics in the same modes as stock.
This is the gate that a value-only A/B comparison cannot provide: a topic
published in the wrong mode carries plausible-looking numbers.
"""
import sys, numpy as np
from pyulog import ULog

TOPICS = ["vehicle_local_position_setpoint", "vehicle_attitude_setpoint",
          "vehicle_rates_setpoint", "vehicle_torque_setpoint",
          "vehicle_thrust_setpoint", "takeoff_status", "rate_ctrl_status"]
# nav_state -> control level
TRAJ = {2, 3, 4, 5, 10, 14, 17, 20, 21, 22}   # POSCTL/AUTO_*/ALTCTL/OFFBOARD/ORBIT
ATT  = {15}                                    # STABILIZED
RATE = {16}                                    # ACRO

def segments(u):
    vs = next(x for x in u.data_list if x.name == "vehicle_status").data
    t = np.asarray(vs['timestamp'], dtype=float)/1e6
    n = np.asarray(vs['nav_state'], dtype=float).astype(int)
    segs, i = [], 0
    while i < len(n):
        j = i
        while j+1 < len(n) and n[j+1] == n[i]:
            j += 1
        if j > i:
            segs.append((n[i], t[i], t[j]))
        i = j+1
    return segs

def level(ns):
    return "Trajectory" if ns in TRAJ else "Attitude" if ns in ATT else "BodyRate" if ns in RATE else f"other({ns})"

def report(path):
    u = ULog(path)
    print(f"\n=== {path.split('/')[-1]}")
    counts = {}
    for tn in TOPICS:
        d = next((x for x in u.data_list if x.name == tn), None)
        counts[tn] = np.asarray(d.data['timestamp'], dtype=float)/1e6 if d else None
    hdr = f"{'level (nav_state)':24s}" + "".join(f"{t.replace('vehicle_','v_')[:15]:>17s}" for t in TOPICS)
    print(hdr)
    rows = {}
    for ns, t0, t1 in segments(u):
        if t1 - t0 < 3.0:
            continue
        lv = level(ns)
        line = f"{lv+' ('+str(ns)+')':24s}"
        for tn in TOPICS:
            ts = counts[tn]
            c = int(((ts >= t0) & (ts <= t1)).sum()) if ts is not None else -1
            line += f"{('n/a' if c < 0 else str(c)):>17s}"
            rows.setdefault((lv, tn), 0)
            rows[(lv, tn)] += max(c, 0)
        print(line)
    return rows

if __name__ == "__main__":
    a = report(sys.argv[1])
    b = report(sys.argv[2])
    print("\n--- DIFFERENCES (topic published in a mode by one build but not the other) ---")
    bad = False
    for k in sorted(set(a) | set(b), key=lambda x: (x[0], x[1])):
        pa, pb = a.get(k, 0), b.get(k, 0)
        if (pa == 0) != (pb == 0):
            print(f"  MISMATCH  {k[0]:12s} {k[1]:34s} stock={pa:5d}  framework={pb:5d}")
            bad = True
    print("  none - publication matrix matches" if not bad else "  ^^^ publication matrix DIFFERS")
