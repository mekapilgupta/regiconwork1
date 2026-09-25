#!/usr/bin/env python3
"""
Master Execution Pipeline: Runs Stages A, B, C, D, and E sequentially.

Guarantees:
- Gate A: All integrity assertions pass, analysis/integrity_report.md generated.
- Gate B: Complete tail failure taxonomy generated, counts sum to total queries.
- Gate C: Title injection evaluated across all channels, deltas computed.
- Gate D: QPP AUROC/AURC metrics and risk-coverage plots generated.
- Gate E: report.md and config.yaml synthesized programmatically with zero forbidden words.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

stages = [
    ("Stage A (Integrity & Channels)", ROOT / "scripts" / "19_stage_a_integrity.py"),
    ("Stage B (Tail Failure Taxonomy)", ROOT / "scripts" / "20_stage_b_tail_taxonomy.py"),
    ("Stage C (Document Title Injection)", ROOT / "scripts" / "21_stage_c_title_injection.py"),
    ("Stage D (QPP & Risk-Coverage)", ROOT / "scripts" / "22_stage_d_qpp_proper.py"),
    ("Stage E (Final Freeze Synthesis)", ROOT / "scripts" / "23_stage_e_final_synthesis.py")
]

def main():
    print("=" * 90)
    print("STARTING FINAL FREEZE EXECUTION PIPELINE (STAGES A - E)")
    print("=" * 90)

    for stage_name, script_path in stages:
        print(f"\n>>> Running {stage_name} ({script_path.name})...")
        cmd = [sys.executable, str(script_path)]
        res = subprocess.run(cmd, cwd=str(ROOT))
        if res.returncode != 0:
            print(f"\n[ERROR] {stage_name} failed with return code {res.returncode}. Aborting.")
            sys.exit(res.returncode)
        print(f">>> {stage_name} COMPLETED SUCCESSFULLY.")

    print("\n" + "=" * 90)
    print("ALL STAGES (A - E) COMPLETED. FINAL FREEZE READY.")
    print("=" * 90)


if __name__ == "__main__":
    main()
