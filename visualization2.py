# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import os

def bar_from_df(df: pd.DataFrame, title: str, out_path: str):
    if df is None or df.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.bar(df["Metric"], df["Value"], color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
