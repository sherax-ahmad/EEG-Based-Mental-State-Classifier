"""
Download EEG datasets from PhysioNet using the wfdb library.

Supported datasets:
  - sleep-edf  : Sleep-EDF Database (sleep staging)
  - eegmmidb   : EEG Motor Movement/Imagery Dataset (focus/relaxed)

Usage:
    python scripts/download_data.py --dataset sleep-edf --n-subjects 20
    python scripts/download_data.py --dataset eegmmidb --n-subjects 10
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import wfdb
except ImportError:
    print("ERROR: wfdb not installed. Run: pip install wfdb")
    sys.exit(1)


RAW_DIR = Path("data/raw")


def download_sleep_edf(n_subjects: int = 20, subset: str = "cassette"):
    """
    Download Sleep-EDF Expanded (cassette or telemetry subset).

    Each subject has:
      - *-PSG.edf        : polysomnographic EEG recording
      - *-Hypnogram.edf  : sleep stage annotations
    """
    out_dir = RAW_DIR / "sleep-edf"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Sleep-EDF ({subset}) — {n_subjects} subjects to {out_dir}")

    db_name = "sleep-edfx"
    try:
        records = wfdb.io.get_record_list(db_name)
    except Exception as e:
        print(f"Could not fetch record list: {e}")
        print("Try: python -m wfdb.io.download -r sleep-edfx")
        return

    # Filter by subset prefix
    prefix = "SC" if subset == "cassette" else "ST"
    filtered = [r for r in records if os.path.basename(r).startswith(prefix)][:n_subjects * 2]

    for rec in filtered:
        try:
            wfdb.dl_files(db_name, str(out_dir), [f"{rec}.edf", f"{rec}.edf.gz"], overwrite=False)
            print(f"  Downloaded: {rec}")
        except Exception as e:
            print(f"  Skipped {rec}: {e}")

    print(f"Done. Files in {out_dir}")


def download_eegmmidb(n_subjects: int = 10):
    """
    Download EEG Motor Movement/Imagery Dataset.

    Each subject has 14 EDF runs (eyes open, eyes closed, motor imagery tasks).
    """
    out_dir = RAW_DIR / "eegmmidb"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading EEGMMIDB — {n_subjects} subjects to {out_dir}")

    db_name = "eegmmidb"
    for subj in range(1, n_subjects + 1):
        subj_dir = out_dir / f"S{subj:03d}"
        subj_dir.mkdir(exist_ok=True)
        try:
            wfdb.dl_database(
                db_name,
                str(subj_dir),
                records=[f"S{subj:03d}/S{subj:03d}R{run:02d}" for run in range(1, 15)],
                overwrite=False,
            )
            print(f"  Subject S{subj:03d} downloaded.")
        except Exception as e:
            print(f"  Subject S{subj:03d} failed: {e}")

    print(f"Done. Files in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download PhysioNet EEG datasets")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["sleep-edf", "eegmmidb"],
        help="Dataset to download"
    )
    parser.add_argument("--n-subjects", type=int, default=20,
                        help="Number of subjects to download")
    parser.add_argument("--subset", type=str, default="cassette",
                        choices=["cassette", "telemetry"],
                        help="Subset for Sleep-EDF (cassette or telemetry)")
    args = parser.parse_args()

    if args.dataset == "sleep-edf":
        download_sleep_edf(n_subjects=args.n_subjects, subset=args.subset)
    elif args.dataset == "eegmmidb":
        download_eegmmidb(n_subjects=args.n_subjects)


if __name__ == "__main__":
    main()
