import os
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def create_sentiment_chart(summary_metrics: dict):
    """
    Creates a bar plot for Positive, Neutral, and Negative sentiment probabilities.
    Returns matplotlib figure.
    """
    pos_pct = summary_metrics.get("avg_positive", 0.0) * 100
    neu_pct = summary_metrics.get("avg_neutral", 0.0) * 100
    neg_pct = summary_metrics.get("avg_negative", 0.0) * 100
    
    categories = ['Positive', 'Neutral', 'Negative']
    values = [pos_pct, neu_pct, neg_pct]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(categories, values, color=colors, height=0.55)
    
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=10, fontweight="bold")
    ax.set_title("Financial News Sentiment Breakdown", fontsize=12, fontweight="bold")
    
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold'
        )
        
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig


def create_plots(df, ticker):

    os.makedirs("../plots", exist_ok=True)

    # ============================
    # 1. Stock Price
    # ============================

    plt.figure(figsize=(12,6))

    plt.plot(
        df["Date"],
        df["Close"],
        linewidth=2,
        label="Close Price"
    )

    plt.title(
        f"{ticker} Closing Price",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    try:
        plt.savefig(
            "../plots/stock_price.png",
            dpi=300
        )
    except Exception:
        pass

    plt.close()

    # ============================
    # 2. Moving Average
    # ============================

    plt.figure(figsize=(12,6))

    plt.plot(
        df["Date"],
        df["Close"],
        linewidth=2,
        label="Close"
    )

    plt.plot(
        df["Date"],
        df["MA20"],
        linewidth=2,
        label="20 Day MA"
    )

    plt.plot(
        df["Date"],
        df["MA50"],
        linewidth=2,
        label="50 Day MA"
    )

    plt.title(
        "Moving Average Strategy",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    try:
        plt.savefig(
            "../plots/moving_average.png",
            dpi=300
        )
    except Exception:
        pass

    plt.close()

    # ============================
    # 3. Buy Sell Signals
    # ============================

    plt.figure(figsize=(14,6))

    plt.plot(
        df["Date"],
        df["Close"],
        linewidth=2,
        label="Close Price"
    )

    buy = df[df["Signal"] == 1]

    sell = df[df["Signal"] == -1]

    plt.scatter(
        buy["Date"],
        buy["Close"],
        marker="^",
        s=120,
        label="BUY"
    )

    plt.scatter(
        sell["Date"],
        sell["Close"],
        marker="v",
        s=120,
        label="SELL"
    )

    plt.title(
        "Trading Signals",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    try:
        plt.savefig(
            "../plots/buy_sell_signals.png",
            dpi=300
        )
    except Exception:
        pass

    plt.close()

    # ============================
    # 4. Portfolio Comparison
    # ============================

    plt.figure(figsize=(12,6))

    plt.plot(
        df["Date"],
        df["Portfolio Value"],
        linewidth=2,
        label="Strategy"
    )

    plt.plot(
        df["Date"],
        df["Buy Hold Value"],
        linewidth=2,
        label="Buy & Hold"
    )

    plt.title(
        "Strategy vs Buy & Hold",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Date")

    plt.ylabel("Portfolio Value")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    try:
        plt.savefig(
            "../plots/cumulative_returns.png",
            dpi=300
        )
    except Exception:
        pass

    plt.close()

    print("\nGraphs generated successfully!")