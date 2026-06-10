from pathlib import Path
import subprocess
import sys
import importlib


# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "outputs" / "data"
RESULT_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

AUTH_FILE = ROOT / "dataset" / "auth" / "auth.txt"
REDTEAM_FILE = ROOT / "dataset" / "redteam.txt"

LABELED = DATA_DIR / "labeled.parquet"
LABELED_SMALL = DATA_DIR / "labeled_small.parquet"

PTG_RESULTS = RESULT_DIR / "ptg_results.json"
BASELINES_RESULTS = RESULT_DIR / "baselines_fair.json"
ABLATION_RESULTS = RESULT_DIR / "weight_ablation.csv"
ROBUSTNESS_RESULTS = RESULT_DIR / "robustness.json"
INTERPRETABILITY_RESULTS = RESULT_DIR / "interpretability.json"


# ==========================================================
# UTILITIES
# ==========================================================

def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_command(name, cmd):
    log_file = LOG_DIR / f"{name}.log"

    banner(f"RUNNING: {name}")
    print(" ".join(map(str, cmd)))

    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        print(f"[FAILED] {name}")
        print(f"Check log: {log_file}")
        sys.exit(1)

    print(f"[OK] {name}")
    print(f"[LOG] {log_file}")


# ==========================================================
# HEALTH CHECK
# ==========================================================

def health_check():
    banner("HEALTH CHECK")

    packages = [
        "pandas",
        "numpy",
        "networkx",
        "sklearn",
        "matplotlib",
        "pyarrow",
    ]

    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"[OK] {pkg}")
        except ImportError:
            print(f"[MISSING] {pkg}")
            sys.exit(1)

    if not AUTH_FILE.exists():
        print(f"[MISSING] {AUTH_FILE}")
        sys.exit(1)

    if not REDTEAM_FILE.exists():
        print(f"[MISSING] {REDTEAM_FILE}")
        sys.exit(1)

    print("[OK] Dataset located")


# ==========================================================
# STATUS
# ==========================================================

def show_status():
    banner("CURRENT STATUS")

    checks = {
        "outputs/data/labeled.parquet": LABELED.exists(),
        "outputs/data/labeled_small.parquet": LABELED_SMALL.exists(),
        "outputs/results/ptg_results.json": PTG_RESULTS.exists(),
        "outputs/results/baselines_fair.json": BASELINES_RESULTS.exists(),
        "outputs/results/weight_ablation.csv": ABLATION_RESULTS.exists(),
        "outputs/results/robustness.json": ROBUSTNESS_RESULTS.exists(),
        "outputs/results/interpretability.json": INTERPRETABILITY_RESULTS.exists(),
        "outputs/figures/": FIGURES_DIR.exists(),
    }

    for name, exists in checks.items():
        mark = "✓" if exists else "✗"
        print(f"{mark} {name}")


# ==========================================================
# STAGES
# ==========================================================

def stage_label(force=False):
    if LABELED.exists() and not force:
        print("[SKIP] labeled.parquet exists")
        return

    run_command(
        "load_label",
        [
            sys.executable,
            str(ROOT / "data_diagnosis" / "load_label.py"),
            "--auth", str(AUTH_FILE),
            "--redteam", str(REDTEAM_FILE),
            "--pad", "86400",
            "--chunksize", "2000000",
            "--out", str(LABELED),
        ],
    )


def stage_subsample(force=False):
    if not LABELED.exists():
        print("[ERROR] Run labeling first")
        return

    if LABELED_SMALL.exists() and not force:
        print("[SKIP] labeled_small.parquet exists")
        return

    run_command(
        "subsample",
        [
            sys.executable,
            str(ROOT / "data_diagnosis" / "subsample.py"),
            "--in", str(LABELED),
            "--out", str(LABELED_SMALL),
            "--neg", "500000",
        ],
    )


def stage_ptg(force=False):
    if not LABELED_SMALL.exists():
        print("[ERROR] Run subsampling first")
        return

    if PTG_RESULTS.exists() and not force:
        print("[SKIP] PTG results exist")
        return

    run_command(
        "ptg_evaluate",
        [
            sys.executable,
            str(ROOT / "core" / "ptg_evaluate.py"),
            "--data", str(LABELED_SMALL),
            "--delta", "3600",
            "--max-depth", "4",
        ],
    )


def stage_baselines(force=False):
    if not LABELED_SMALL.exists():
        print("[ERROR] Run subsampling first")
        return

    if BASELINES_RESULTS.exists() and not force:
        print("[SKIP] Baselines exist")
        return

    run_command(
        "baselines_fair",
        [
            sys.executable,
            str(ROOT / "experiments" / "baselines_fair.py"),
            "--data", str(LABELED_SMALL),
            "--delta", "3600",
        ],
    )


def stage_ablation(force=False):
    if not LABELED_SMALL.exists():
        print("[ERROR] Run subsampling first")
        return

    if ABLATION_RESULTS.exists() and not force:
        print("[SKIP] Ablation exists")
        return

    run_command(
        "ablation",
        [
            sys.executable,
            str(ROOT / "experiments" / "ablation.py"),
            "--data", str(LABELED_SMALL),
            "--max-depth", "4",
            "--delta", "3600",
        ],
    )


def stage_robustness(force=False):
    if not LABELED_SMALL.exists():
        print("[ERROR] Run subsampling first")
        return

    script = ROOT / "experiments" / "robustness.py"
    if not script.exists():
        print("[INFO] robustness.py not found")
        return

    if ROBUSTNESS_RESULTS.exists() and not force:
        print("[SKIP] Robustness exists")
        return

    run_command(
        "robustness",
        [
            sys.executable,
            str(script),
            "--small", str(LABELED_SMALL),
            "--seeds", "5",
            "--delta", "3600",
            "--max-depth", "4",
            "--alpha", "0.3",
            "--beta", "0.5",
            "--gamma", "0.2",
        ],
    )


def stage_interpretability(force=False):
    if not LABELED_SMALL.exists():
        print("[ERROR] Run subsampling first")
        return

    script = ROOT / "experiments" / "interpretability.py"
    if not script.exists():
        print("[INFO] interpretability.py not found")
        return

    if INTERPRETABILITY_RESULTS.exists() and not force:
        print("[SKIP] Interpretability results exist")
        return

    run_command(
        "interpretability",
        [
            sys.executable,
            str(script),
            "--data", str(LABELED_SMALL),
            "--delta", "3600",
            "--max-depth", "4",
            "--K", "25",
        ],
    )


def stage_figures(force=False):
    script = ROOT / "core" / "make_figures.py"
    if not script.exists():
        print("[INFO] make_figures.py not found")
        return

    run_command(
        "make_figures",
        [
            sys.executable,
            str(script),
        ],
    )


# ==========================================================
# PIPELINE
# ==========================================================

def run_pipeline(force=False):
    health_check()

    stage_label(force)
    stage_subsample(force)

    stage_ptg(force)
    stage_baselines(force)
    stage_ablation(force)
    stage_robustness(force)
    stage_interpretability(force)
    stage_figures(force)

    banner("PIPELINE COMPLETE")


# ==========================================================
# MENU
# ==========================================================

def main():
    while True:
        print("""
==================================================
PTG FRAMEWORK
==================================================

1. Show Status
2. Run Pipeline
3. Force Rebuild
4. Run Label Stage
5. Run Subsample Stage
6. Run PTG Evaluation
7. Run Fair Baselines
8. Run Ablation
9. Run Robustness
10. Run Interpretability
11. Generate Figures
0. Exit
""")

        choice = input("Choice: ").strip()

        if choice == "1":
            show_status()
        elif choice == "2":
            run_pipeline(False)
        elif choice == "3":
            run_pipeline(True)
        elif choice == "4":
            health_check()
            stage_label(False)
        elif choice == "5":
            stage_subsample(False)
        elif choice == "6":
            stage_ptg(False)
        elif choice == "7":
            stage_baselines(False)
        elif choice == "8":
            stage_ablation(False)
        elif choice == "9":
            stage_robustness(False)
        elif choice == "10":
            stage_interpretability(False)
        elif choice == "11":
            stage_figures(False)
        elif choice == "0":
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()