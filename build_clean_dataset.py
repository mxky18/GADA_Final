

import pandas as pd
from pathlib import Path

MASTER = Path(__file__).parent / "master_features.csv"
OUT    = Path(__file__).parent / "master_features_clean.csv"

MIN_DURATION_SEC = 10

DROP_TRANSITIONS = {
    ("AEROBIC",   "Warm-up"),
    ("AEROBIC",   "Baseline + Warm-up"),
    ("AEROBIC",   "Cool Down"),
    ("ANAEROBIC", "Warm-up"),
    ("ANAEROBIC", "Cool Down 1"),
    ("ANAEROBIC", "Cool Down 2"),
    ("ANAEROBIC", "Cool Down 3"),
    ("ANAEROBIC", "Cool Down 4"),
}


def main():
    df = pd.read_csv(MASTER, sep=";")
    print(f"Original: {len(df)} Zeilen")
    print("Label-Verteilung vorher:")
    print(df["label"].value_counts().to_string())

    is_transition = df.apply(
        lambda r: r["label"] == "rest"
        and (r["category"], r["phase_name"]) in DROP_TRANSITIONS,
        axis=1,
    )
    print(f"\nEntferne {is_transition.sum()} aktive-Transition-rest-Bloecke")

    is_short = df["duration_sec"] < MIN_DURATION_SEC
    print(f"Entferne {is_short.sum()} zu kurze Bloecke (duration_sec < {MIN_DURATION_SEC}):")
    print(df.loc[is_short, ["subject", "category", "phase_name", "label", "duration_sec"]]
          .to_string(index=False))

    clean = df[~(is_transition | is_short)].reset_index(drop=True)
    clean.to_csv(OUT, index=False, sep=";")

    print(f"\nBereinigt: {len(clean)} Zeilen -> {OUT.name}")
    print("Label-Verteilung nachher:")
    print(clean["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
