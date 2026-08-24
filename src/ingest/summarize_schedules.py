"""Print coverage and null stats for the cached schedules parquet."""

import pandas as pd

from src.ingest.nflverse import RAW_DIR


def main() -> None:
    df = pd.read_parquet(RAW_DIR / "schedules.parquet")

    print("Rows per season:")
    print(df.groupby("season").size().to_string())

    print("\nNull counts:")
    print(f"  spread_line: {df['spread_line'].isna().sum()}")
    print(f"  total_line:  {df['total_line'].isna().sum()}")

    populated = df.loc[df["spread_line"].notna(), "season"]
    earliest = populated.min() if not populated.empty else None
    print(f"\nEarliest season with spread_line populated: {earliest}")


if __name__ == "__main__":
    main()
