#!/usr/bin/env python
"""
test_color.py - validate that the E57 color actually loads.

Checks three reader variants on the SAME file and reports which return real color:
  A) read_scan(ignore_missing_fields=True)              # our old (broken) call
  B) read_scan(colors=True, intensity=True, ...)        # the SSL-script call (known good)
  C) read_scan_raw(i)                                   # raw-vector call
Then runs the project's geometry.load_e57 and confirms has_colors + color range.

Usage: python test_color.py /path/to/scan.e57
"""
import sys, numpy as np

def summarize(tag, d):
    keys = list(d.keys())
    has = all(k in d for k in ("colorRed", "colorGreen", "colorBlue"))
    line = f"[{tag}] keys={keys[:4]}{'...' if len(keys)>4 else ''} | color_fields={has}"
    if has:
        c = np.column_stack([np.asarray(d["colorRed"], np.float64),
                             np.asarray(d["colorGreen"], np.float64),
                             np.asarray(d["colorBlue"], np.float64)])
        line += f" | n={len(c)} min={c.min():.1f} max={c.max():.1f} mean={c.mean():.1f}"
        line += " | REAL COLOR" if c.size and c.max() > 0 else " | all-zero (no real color)"
    print(line)
    return has

def main():
    if len(sys.argv) < 2:
        print("usage: python test_color.py /path/to/scan.e57"); return
    path = sys.argv[1]
    import pye57
    e = pye57.E57(path)
    print(f"file: {path} | scan_count={e.scan_count}")
    print("header fields:", e.get_header(0).point_fields, "\n")

    # A) old call (no flags)
    try:
        summarize("A read_scan()             ", e.read_scan(0, ignore_missing_fields=True))
    except Exception as ex:
        print("[A] failed:", ex)

    # B) SSL-script call (colors=True, intensity=True)  <-- expected GOOD
    try:
        summarize("B read_scan(colors=True)  ",
                  e.read_scan(0, colors=True, intensity=True, ignore_missing_fields=True))
    except Exception as ex:
        print("[B] failed:", ex)

    # C) raw-vector call
    try:
        summarize("C read_scan_raw()         ", e.read_scan_raw(0))
    except Exception as ex:
        print("[C] failed:", ex)

    # D) end-to-end through the project loader
    print("\n--- project geometry.load_e57 ---")
    try:
        import geometry as G
        full = G.load_e57(path)
        ok = full.has_colors()
        print(f"[D] load_e57: has_colors={ok}")
        if ok:
            cols = np.asarray(full.colors)
            print(f"    color range [{cols.min():.3f}, {cols.max():.3f}], "
                  f"distinct≈{len(np.unique((cols*255).astype(int).reshape(-1,3),axis=0))}")
            print("    PASS: color loads correctly.")
        else:
            print("    FAIL: loader returned no color. Fix load_e57 to use the variant that "
                  "printed 'REAL COLOR' above (B or C).")
    except Exception as ex:
        print("[D] load_e57 error:", ex)

if __name__ == "__main__":
    main()
