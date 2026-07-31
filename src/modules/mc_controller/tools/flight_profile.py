#!/usr/bin/env python3
"""
Deterministic SITL flight profile for the mc_controller A/B.

Flown identically with MC_CTRL_ALG=0 (stock) and MC_CTRL_ALG=1 (reference
cascade); the resulting ULogs are compared phase-aligned.

Uses OFFBOARD position setpoints rather than manual sticks for the bulk of the
flight so the trajectory is repeatable between runs. An optional STABILIZED
segment (PROFILE_STAB=1) drives MANUAL_CONTROL to exercise the Attitude level,
which the OFFBOARD-only profile never reaches.

Two environment facts encoded here, both discovered the hard way:
  1. Headless SITL will not arm without something heartbeating as a GCS
     (Preflight Fail: No connection to the GCS).
  2. MAV_CMD_NAV_TAKEOFF is ACKed but does not climb (goes to AUTO.LOITER and
     auto-disarms); OFFBOARD is entered BEFORE arming instead.
"""
import math
import os
import sys
import threading
import time

from pymavlink import mavutil

TAKEOFF_ALT = 5.0
SQUARE = 6.0
LEG_HOLD = 8.0
CONN = "udp:127.0.0.1:14540"
STABILIZED_SEGMENT = os.environ.get("PROFILE_STAB", "0") == "1"

_stop_hb = False


def log(msg):
    print(f"[profile] {msg}", flush=True)


def start_gcs_heartbeat(m):
    def _hb():
        while not _stop_hb:
            m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                 mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            time.sleep(1.0)

    threading.Thread(target=_hb, daemon=True).start()


def wait_ready(m, timeout=180):
    log("waiting for EKF / home position...")
    deadline = time.time() + timeout
    got_fix = False
    while time.time() < deadline:
        msg = m.recv_match(type=["LOCAL_POSITION_NED", "GPS_RAW_INT"], blocking=True, timeout=5)
        if msg is None:
            continue
        if msg.get_type() == "GPS_RAW_INT" and msg.fix_type >= 3:
            got_fix = True
        if got_fix and msg.get_type() == "LOCAL_POSITION_NED":
            log("EKF ready")
            time.sleep(5)
            return True
    log("TIMEOUT waiting for EKF")
    return False


def cmd(m, command, *params):
    m.mav.command_long_send(m.target_system, m.target_component, command, 0,
                            *(list(params) + [0] * (7 - len(params))))
    return m.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)


def arm(m):
    log("arming...")
    ack = None
    for _ in range(10):
        ack = cmd(m, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        if ack and ack.result == 0:
            log("armed")
            return True
        time.sleep(2)
    log(f"ARM FAILED (last ack: {ack})")
    return False


def send_setpoint(m, n, e, d, yaw):
    m.mav.set_position_target_local_ned_send(
        int(time.time() * 1e3) & 0xFFFFFFFF,
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000101111111000, n, e, d, 0, 0, 0, 0, 0, 0, yaw, 0)


def offboard_leg(m, n, e, d, yaw, hold):
    end = time.time() + hold
    while time.time() < end:
        send_setpoint(m, n, e, d, yaw)
        time.sleep(0.05)


def enter_offboard(m, n, e, d, yaw):
    for _ in range(40):
        send_setpoint(m, n, e, d, yaw)
        time.sleep(0.05)
    log("switching to OFFBOARD...")
    ack = cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE,
              mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
    if not ack or ack.result != 0:
        log(f"OFFBOARD REJECTED: {ack}")
        return False
    log("in OFFBOARD")
    return True


def climb_offboard(m, alt, timeout=90):
    log(f"climbing to {alt} m under OFFBOARD...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        send_setpoint(m, 0, 0, -alt, 0.0)
        msg = m.recv_match(type="LOCAL_POSITION_NED", blocking=False)
        if msg and -msg.z >= alt * 0.92:
            log(f"reached {-msg.z:.2f} m")
            time.sleep(3)
            return True
        time.sleep(0.05)
    log("TIMEOUT during climb")
    return False


def stabilized_stick_sweep(m, duration=12.0):
    """Exercise StickToAttitudeSetpoint in situ (Attitude level).

    The OFFBOARD-only profile never reaches this path, which is the Stage 1
    coverage gap. Requires COM_RC_IN_MODE to accept MANUAL_CONTROL.
    """
    # PX4 will not accept a manual mode without a live manual-control stream, and
    # the DO_SET_MODE is ACKed even when the switch does not take. Establish the
    # stream FIRST, then switch, then verify via HEARTBEAT that it actually took.
    log("establishing MANUAL_CONTROL stream before mode switch...")
    for _ in range(40):
        m.mav.manual_control_send(m.target_system, 0, 0, 600, 0, 0)
        time.sleep(0.05)

    log("switching to STABILIZED for stick sweep...")
    cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 7, 0)

    # Verify the mode actually engaged: custom_mode main byte 7 == STABILIZED.
    engaged = False
    deadline = time.time() + 5
    while time.time() < deadline:
        m.mav.manual_control_send(m.target_system, 0, 0, 600, 0, 0)
        hb = m.recv_match(type="HEARTBEAT", blocking=False)
        if hb and ((hb.custom_mode >> 16) & 0xFF) == 7:
            engaged = True
            break
        time.sleep(0.05)

    if not engaged:
        log("STABILIZED did NOT engage (ACK is not proof) - skipping stick sweep")
        return False

    log("STABILIZED engaged")

    end = time.time() + duration
    t0 = time.time()
    while time.time() < end:
        phase = (time.time() - t0) / duration
        x = int(300 * math.sin(2 * math.pi * phase))   # pitch
        y = int(300 * math.cos(2 * math.pi * phase))   # roll
        z = 600                                        # throttle 0..1000
        r = int(150 * math.sin(4 * math.pi * phase))   # yaw
        m.mav.manual_control_send(m.target_system, x, y, z, r, 0)
        time.sleep(0.05)

    log("stick sweep complete")
    return True


def land(m):
    log("landing...")
    cmd(m, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, float("nan"), 0, 0, 0)
    deadline = time.time() + 120
    while time.time() < deadline:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=5)
        if hb and not (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            log("disarmed after landing")
            return True
    log("TIMEOUT waiting for disarm")
    return False


def main():
    global _stop_hb
    m = mavutil.mavlink_connection(CONN)
    log("waiting for heartbeat...")
    m.wait_heartbeat(timeout=60)
    log(f"heartbeat: sys={m.target_system} comp={m.target_component}")
    start_gcs_heartbeat(m)

    if not wait_ready(m):
        return 1

    d = -TAKEOFF_ALT
    if not enter_offboard(m, 0, 0, d, 0.0):
        return 1
    if not arm(m):
        return 1
    if not climb_offboard(m, TAKEOFF_ALT):
        return 1

    log("flying square...")
    for (n, e) in [(SQUARE, 0), (SQUARE, SQUARE), (0, SQUARE), (0, 0)]:
        log(f"  leg -> N={n} E={e}")
        offboard_leg(m, n, e, d, 0.0, LEG_HOLD)

    log("yaw sweep in place...")
    for yaw_deg in (45, 90, 45, 0):
        offboard_leg(m, 0, 0, d, math.radians(yaw_deg), 5.0)

    if STABILIZED_SEGMENT:
        stabilized_stick_sweep(m)
        log("returning to OFFBOARD...")
        enter_offboard(m, 0, 0, d, 0.0)
        offboard_leg(m, 0, 0, d, 0.0, 5.0)

    if not land(m):
        return 1

    _stop_hb = True
    log("PROFILE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
