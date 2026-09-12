import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ktp_register import mask_nik, register_from_ktp


def load_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            rows.append(row)
    return rows


def load_state(state_path):
    if not os.path.isfile(state_path):
        return set()
    with open(state_path, encoding="utf-8-sig", newline="") as f:
        return {r["filename"] for r in csv.DictReader(f, delimiter=";")}


def save_state(state_path, files):
    with open(state_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["filename", "nik"])
        for line in files:
            w.writerow(line)


def main():
    ap = argparse.ArgumentParser(
        description="Register all KTP rows in master CSV (batch, skips already imported)")
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "master_ktp.csv"))
    ap.add_argument("--ktps", default=os.path.join(ROOT, "ktps"))
    ap.add_argument("--report", default=os.path.join(ROOT, "data", "batch_report.csv"))
    ap.add_argument("--state", default=os.path.join(ROOT, "data", "batch_state.csv"))
    ap.add_argument("--force", action="store_true", help="re-register even if already imported")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    imported = load_state(args.state)
    report = []
    state_lines = []

    for row in rows:
        filename = row.get("filename", "").strip()
        nik = row.get("nik", "").strip()
        nama = row.get("nama", "").strip()
        path = os.path.join(args.ktps, filename)
        masked = mask_nik(nik) if len(nik) == 16 else nik

        if filename in imported and not args.force:
            report.append((filename, masked, "skip", "already imported"))
            state_lines.append((filename, nik))
            print(f"[SKIP] {masked:15} {nama:20} {filename}  (already imported)")
            continue

        try:
            ok = register_from_ktp(nik, path, name=nama or None)
            status, msg = ("ok", "registered") if ok else ("fail", "registration returned False")
        except Exception as e:
            status, msg = "fail", f"{type(e).__name__}: {e}"
        report.append((filename, masked, status, msg))
        print(f"[{status.upper():4}] {masked:15} {nama:20} {filename}  ({msg})")
        if status == "ok":
            state_lines.append((filename, nik))

    save_state(args.state, state_lines)

    ok_count = sum(1 for r in report if r[2] == "ok")
    with open(args.report, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["filename", "nik", "status", "pesan"])
        w.writerows(report)

    print(f"\nsummary: {ok_count} ok / {len(report)} rows -> {args.report}")


if __name__ == "__main__":
    main()