import warnings
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
from tabpfn_multiclass import run_variant, print_summary, make_comparison_figure

warnings.filterwarnings("ignore", message=".*Running on CPU with more than 200 samples.*")

DATA_PATH = Path(__file__).parent / "master_features_clean.csv"


def main():
    df = pd.read_csv(DATA_PATH, sep=";")
    print(f"Lade {len(df)} Zeilen aus master_features_clean.csv ...")
    print(f"Klassenverteilung: {df['label'].value_counts().to_dict()}")
    print("TabPFN LOSOCV (CPU) - bitte etwas Geduld ...\n")

    variants = [run_variant(df, include_acc=True),
                run_variant(df, include_acc=False)]
    print_summary(variants)

    for v in variants:
        print(f"\n=== Classification Report – {v['name']} ===")
        print(classification_report(v["y"], v["y_pred"],
                                     target_names=list(v["le"].classes_), digits=3))

    make_comparison_figure(variants, Path(__file__).parent / "tabpfn_multiclass_clean_results.png")


if __name__ == "__main__":
    main()
