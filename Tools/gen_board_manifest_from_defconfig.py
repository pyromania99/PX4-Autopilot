#!/usr/bin/env python3
import argparse, json, re, sys

def parse_defconfig(path: str):
    d = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or not line.startswith("CONFIG_"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            d[k] = v
    return d

def norm_hex(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if s.lower().startswith("0x"):
        return s.lower()
    try:
        return "0x{:04x}".format(int(s, 0))
    except Exception:
        return s

def detect_chip(defcfg: dict) -> str:
    # Exact: find CONFIG_ARCH_CHIP_* = y
    for k, v in defcfg.items():
        if k.startswith("CONFIG_ARCH_CHIP_") and v == "y":
            return k[len("CONFIG_ARCH_CHIP_"):].lower().replace("_", "")
    # If the tree has CONFIG_ARCH_CHIP as a string, accept it as-is
    s = defcfg.get("CONFIG_ARCH_CHIP", "")
    return s.lower().replace("_", "") if s else ""

def main():
    ap = argparse.ArgumentParser(description="Strict manifest fragment from defconfig, print to stdout")
    ap.add_argument("--defconfig", required=True)
    args = ap.parse_args()

    d = parse_defconfig(args.defconfig)
    if not d:
        print(f"error: no CONFIG_* entries in {args.defconfig}", file=sys.stderr)
        return 2

    arch = d.get("CONFIG_ARCH", "").lower()
    chip = detect_chip(d)

    vid  = norm_hex(d.get("CONFIG_CDCACM_VENDORID", ""))
    pid  = norm_hex(d.get("CONFIG_CDCACM_PRODUCTID", ""))

    manufacturer = d.get("CONFIG_CDCACM_VENDORSTR", "")
    productstr   = d.get("CONFIG_CDCACM_PRODUCTSTR", "")

    out = {
        "manufacturer": manufacturer,
        "hardware": {
            "architecture": arch,
            "vendor_id": vid,
            "product_id": pid,
            "chip": chip,
            "productstr": productstr
        }
    }

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
