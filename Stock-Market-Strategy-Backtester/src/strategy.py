import numpy as np


def generate_signals(df):

    """
    Moving Average Crossover Strategy
    """

    df = df.copy()

    df["Signal"] = 0

    buy = df["MA20"] > df["MA50"]

    sell = df["MA20"] < df["MA50"]

    df.loc[buy, "Signal"] = 1

    df.loc[sell, "Signal"] = -1

    df["Position"] = df["Signal"].shift(1)

    df["Position"] = df["Position"].fillna(0)

    signal_map = {
        1: "BUY",
        -1: "SELL",
        0: "HOLD"
    }

    df["Action"] = df["Signal"].map(signal_map)

    return df