"""Run the whole pipeline end to end (raw TSVs -> output/matching_results.tsv + candidate_pairs.tsv).

Each step is a separate process (memory is returned to the OS between steps) and the run stops at
the first failing step. Everything is logged to <work>/pipeline_log.txt.

Run (from code/business_entity_resolution/):
    python src/run_pipeline.py                 # all steps
    python src/run_pipeline.py --from-step 5   # resume after a crash (earlier outputs are reused)
    python src/run_pipeline.py --workers 6     # fewer cores / less RAM
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
for _v in ("ER_DATA_DIR", "ER_WORK_DIR", "ER_OUTPUT_DIR"):   # steps run with cwd=src: make paths absolute
    if os.environ.get(_v):
        os.environ[_v] = str(Path(os.environ[_v]).resolve())
import config  # noqa: E402


def steps(w):
    py = sys.executable
    return [
        ("mine dictionaries (train fold)", [py, SRC / "mine_dictionaries.py", "--workers", w]),
        ("clean all 6 source files", [py, SRC / "clean_data.py", "--workers", w]),
        ("blocking: train (all S1 queries)", [py, SRC / "blocking.py", "--split", "train", "--mode", "all"]),
        ("blocking: test", [py, SRC / "blocking.py", "--split", "test"]),
        ("pair features: train", [py, SRC / "features.py", "--split", "train", "--workers", w]),
        ("pair features: test", [py, SRC / "features.py", "--split", "test", "--workers", w]),
        ("train matcher + tune threshold", [py, SRC / "train_matcher.py", "--threads", w]),
        ("predict test + write output", [py, SRC / "predict.py", "--threads", w]),
        ("validate submission files", [py, config.RESOURCE_DIR / "utils" / "validate_submission.py",
                                       "--matching", config.OUTPUT_DIR / "matching_results.tsv",
                                       "--candidate", config.OUTPUT_DIR / "candidate_pairs.tsv",
                                       "--test-dir", config.DATA_DIR / "test"]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--to-step", type=int, default=99)
    args = ap.parse_args()
    env = dict(os.environ, NUMBA_NUM_THREADS=str(args.workers), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    log_path = config.WORK_DIR / "pipeline_log.txt"
    all_steps = steps(str(args.workers))
    t_all = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        def emit(msg):
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

        emit(f"\n===== pipeline start {time.strftime('%Y-%m-%d %H:%M:%S')} | data={config.DATA_DIR} "
             f"work={config.WORK_DIR} output={config.OUTPUT_DIR} workers={args.workers}")
        for i, (name, cmd) in enumerate(all_steps, 1):
            if i < args.from_step or i > args.to_step:
                continue
            emit(f"\n--- step {i}/{len(all_steps)}: {name}")
            t0 = time.time()
            proc = subprocess.Popen([str(c) for c in cmd], cwd=SRC, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                line = line.rstrip()
                if "NumbaWarning" in line or "warnings.warn" in line:
                    continue
                emit("    " + line)
            proc.wait()
            if proc.returncode != 0:
                emit(f"!!! step {i} FAILED (exit {proc.returncode}) after {time.time()-t0:.0f}s. "
                     f"Fix the error above, then resume with:  python src/run_pipeline.py --from-step {i}")
                sys.exit(proc.returncode)
            emit(f"--- step {i} ok ({time.time()-t0:.0f}s)")
        emit(f"\n===== pipeline finished in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
