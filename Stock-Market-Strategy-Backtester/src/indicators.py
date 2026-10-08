import pandas as pd


def calculate_indicators(df: pd.DataFrame):

    """
    Calculates technical indicators.
    """

    df = df.copy()

    df["MA20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["MA50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )

    df["Daily Return"] = (
        df["Close"]
        .pct_change()
        .fillna(0)
    )

    return df