from data_feed import DataFeed
from engine import BacktestEngine
from evaluator import Evaluator
from strategy import Strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    data_path = "validation_f.parquet"

    try:
        print("Loading data...")
        feed = DataFeed(data_path)

        print("Initializing strategy...")
        strategy = Strategy()

        engine = BacktestEngine(data_feed=feed, strategy=strategy)

        # Run backtest and capture returns
        portfolio_returns = engine.run()

        # Save returns to CSV for later analysis
        portfolio_returns.to_csv("portfolio_returns.csv")
        print("Portfolio returns saved to portfolio_returns.csv")

        # Basic evaluation
        evaluator = Evaluator(portfolio_returns, periods_per_year=252 * 78)
        evaluator.generate_report()

        # Generate additional plots for report
        generate_report_figures(portfolio_returns)

    except Exception as e:
        print(f"\n[BACKTEST FAILED] {type(e).__name__}: {e}")


def generate_report_figures(returns):
    """Generate plots for the report."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    # Cumulative wealth (from 5-min returns)
    cumulative = (1 + returns).cumprod()

    # Aggregate 5-min returns to daily returns (every 78 periods = 1 day)
    # Resample using number of periods instead of datetime
    n_periods = len(returns)
    n_days = n_periods // 78
    daily_returns = []
    for i in range(n_days):
        start_idx = i * 78
        end_idx = start_idx + 78
        # Calculate daily return as product of (1 + returns)
        day_ret = (1 + returns.iloc[start_idx:end_idx]).prod() - 1
        daily_returns.append(day_ret)
    daily_returns = pd.Series(daily_returns)

    # Rolling Sharpe on daily returns (20-day window)
    rolling_sharpe = daily_returns.rolling(20).mean() / daily_returns.rolling(20).std() * np.sqrt(252)

    # Drawdown (on cumulative wealth from 5-min returns)
    cumulative_wealth = (1 + returns).cumprod()
    rolling_max = cumulative_wealth.cummax()
    drawdown = (cumulative_wealth - rolling_max) / rolling_max

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Cumulative return
    axes[0].plot(range(len(cumulative)), cumulative.values)
    axes[0].set_title('Cumulative Portfolio Return')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].set_xlabel('5-min Periods')
    axes[0].grid(True)

    # Rolling Sharpe
    axes[1].plot(range(len(rolling_sharpe)), rolling_sharpe.values)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_title('Rolling 20-Day Sharpe Ratio (Daily Aggregated)')
    axes[1].set_ylabel('Sharpe Ratio')
    axes[1].set_xlabel('Trading Days')
    axes[1].grid(True)

    # Drawdown
    axes[2].fill_between(range(len(drawdown)), drawdown.values * 100, 0, color='red', alpha=0.3)
    axes[2].set_title('Drawdown')
    axes[2].set_ylabel('Drawdown (%)')
    axes[2].set_xlabel('5-min Periods')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('strategy_performance.png', dpi=150, bbox_inches='tight')
    print("Performance plot saved to strategy_performance.png")

    # Additional statistics
    print("\n--- Additional Performance Statistics ---")
    print(f"Total 5-min Periods: {len(returns)}")
    print(f"Number of Trading Days: {len(daily_returns)}")
    print(f"Positive 5-min Periods: {(returns > 0).sum()} ({100 * (returns > 0).sum() / len(returns):.1f}%)")
    print(f"Negative 5-min Periods: {(returns < 0).sum()} ({100 * (returns < 0).sum() / len(returns):.1f}%)")
    print(f"Average 5-min Return: {returns.mean():.6f}")
    print(f"5-min Return Std: {returns.std():.6f}")
    print(f"Skewness: {returns.skew():.4f}")
    print(f"Kurtosis: {returns.kurtosis():.4f}")
    print(f"Daily Return Mean: {daily_returns.mean():.6f}")
    print(f"Daily Return Std: {daily_returns.std():.6f}")


if __name__ == "__main__":
    main()