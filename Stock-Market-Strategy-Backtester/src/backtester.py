import os
import time


def backtest_strategy(df, initial_capital=100000):
    """
    Backtests the Moving Average Crossover Strategy.
    """

    start = time.time()

    df = df.copy()

    # -----------------------------
    # Market Return
    # -----------------------------
    df["Market Return"] = (
        df["Close"]
        .pct_change()
        .fillna(0)
    )

    # -----------------------------
    # Strategy Return
    # -----------------------------
    df["Strategy Return"] = (
        df["Position"] *
        df["Market Return"]
    )

    # -----------------------------
    # Portfolio Value
    # -----------------------------
    df["Portfolio Value"] = (
        (1 + df["Strategy Return"])
        .cumprod()
        * initial_capital
    )

    # -----------------------------
    # Buy & Hold
    # -----------------------------
    df["Buy Hold Value"] = (
        (1 + df["Market Return"])
        .cumprod()
        * initial_capital
    )

    # -----------------------------
    # Metrics
    # -----------------------------
    strategy_return = (
        (
            df["Portfolio Value"].iloc[-1]
            - initial_capital
        )
        / initial_capital
    ) * 100

    buy_hold_return = (
        (
            df["Buy Hold Value"].iloc[-1]
            - initial_capital
        )
        / initial_capital
    ) * 100

    outperformance = (
        strategy_return
        - buy_hold_return
    )

    total_signals = (
        df["Signal"]
        .diff()
        .fillna(0)
        .ne(0)
        .sum()
    )

    execution_time = (
        time.time() - start
    )

    # -----------------------------
    # Save Summary
    # -----------------------------
    os.makedirs("../results", exist_ok=True)

    summary = f"""
=========================================
PERFORMANCE SUMMARY
=========================================

Initial Capital : Rs. {initial_capital:,.2f}

Final Portfolio : Rs. {df['Portfolio Value'].iloc[-1]:,.2f}

Strategy Return : {strategy_return:.2f} %

Buy & Hold Return : {buy_hold_return:.2f} %

Outperformance : {outperformance:.2f} %

Signals Generated : {total_signals}

Execution Time : {execution_time:.2f} sec
"""

    with open(
        "../results/performance_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(summary)

    print(summary)

    return df