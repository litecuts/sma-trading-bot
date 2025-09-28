
"""
sma_trading_bot_single.py
A self-contained SMA crossover backtest that works out-of-the-box.
- By default it generates synthetic OHLCV data (so no CSV needed).
- Optionally pass --csv path/to/file.csv with columns: date,open,high,low,close,volume
Outputs:
- outputs/metrics.json
- outputs/trade_log.csv
- outputs/price_signals.png
- outputs/equity_curve.png
Usage:
  python sma_trading_bot_single.py
  python sma_trading_bot_single.py --csv my_data.csv --fast 10 --slow 40 --fee_bps 3
"""
import argparse, os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_synthetic_prices(start_date="2023-01-01", days=500, s0=400.0, mu=0.10, sigma=0.20, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * (dt**0.5), size=days)
    prices = [s0]
    for eps in shocks:
        prices.append(prices[-1] * math.exp(eps))
    dates = pd.bdate_range(start=start_date, periods=days+1)
    df = pd.DataFrame({"date": dates, "close": prices})
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open","close"]].max(axis=1) * (1 + 0.002)
    df["low"]  = df[["open","close"]].min(axis=1) * (1 - 0.002)
    df["volume"] = (1e6 + np.abs(rng.normal(0, 2e5, size=len(df)))).astype(int)
    return df

def backtest_sma(df, fast=20, slow=50, fee_bps=5, initial_cash=10000.0):
    df = df.copy()
    # Ensure date column is string or datetime
    if "date" not in df.columns:
        raise ValueError("CSV must include a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["close"]:
        if col not in df.columns:
            raise ValueError(f"CSV must include '{col}' column.")
    df = df.sort_values("date").reset_index(drop=True)

    df["fast_sma"] = df["close"].rolling(fast).mean()
    df["slow_sma"] = df["close"].rolling(slow).mean()
    df["signal"] = (df["fast_sma"] > df["slow_sma"]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0)
    df["trade"] = df["position"].diff().fillna(df["position"])
    df["ret"] = df["close"].pct_change().fillna(0.0)

    equity = [initial_cash]
    cash = initial_cash
    trade_rows = []

    for i in range(1, len(df)):
        prev_eq = equity[-1]
        trade = df.loc[i, "trade"]
        if trade != 0:
            fee = prev_eq * (fee_bps/10000.0)
            cash -= fee
            trade_rows.append({
                "date": str(df.loc[i, "date"].date()),
                "action": "BUY" if trade > 0 else "SELL",
                "fee": round(fee, 2),
                "equity_before": round(prev_eq, 2)
            })
            prev_eq = cash

        position_flag = int(df.loc[i-1, "position"])
        daily_ret = df.loc[i, "ret"]
        if position_flag == 1:
            new_eq = prev_eq * (1 + daily_ret)
        else:
            new_eq = prev_eq
        cash = new_eq
        equity.append(new_eq)

    df["equity"] = equity
    df["equity_curve"] = df["equity"] / initial_cash

    roll_max = df["equity_curve"].cummax()
    drawdown = df["equity_curve"] / roll_max - 1

    strat_daily = df["equity_curve"].pct_change().fillna(0.0)
    vol_annual = strat_daily.std() * (252**0.5)
    sharpe = (strat_daily.mean() * 252) / (vol_annual + 1e-12)

    total_return = df["equity"].iloc[-1] / initial_cash - 1
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = max(days/365.25, 1e-9)
    cagr = (1 + total_return) ** (1/years) - 1

    # round-trip trades win-rate
    trades = []
    in_trade = False
    entry_equity = None
    for i in range(len(df)):
        if not in_trade and df["trade"].iloc[i] > 0:
            in_trade = True
            entry_equity = df["equity"].iloc[i]
        elif in_trade and df["trade"].iloc[i] < 0:
            exit_equity = df["equity"].iloc[i]
            trades.append(exit_equity / entry_equity - 1)
            in_trade = False
    num_trades = len(trades)
    win_rate = (np.array(trades) > 0).mean() if trades else 0.0
    avg_trade_ret = float(np.mean(trades)) if trades else 0.0

    metrics = {
        "fast_sma": fast, "slow_sma": slow, "fee_bps": fee_bps,
        "initial_cash": initial_cash,
        "total_return_pct": round(total_return*100, 2),
        "CAGR_pct": round(cagr*100, 2),
        "annual_volatility_pct": round(vol_annual*100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(float(drawdown.min())*100, 2),
        "num_trades": int(num_trades),
        "win_rate_pct": round(float(win_rate)*100, 2),
        "avg_trade_return_pct": round(avg_trade_ret*100, 2),
        "backtest_days": int(days)
    }
    return df, metrics, pd.DataFrame(trade_rows)

def save_outputs(df, metrics, trade_log, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    # Metrics
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # Trade log
    trade_log.to_csv(os.path.join(out_dir, "trade_log.csv"), index=False)

    # Price + SMA + signals
    plt.figure(figsize=(10,6))
    plt.plot(df["date"], df["close"], label="Close")
    plt.plot(df["date"], df["fast_sma"], label="Fast SMA")
    plt.plot(df["date"], df["slow_sma"], label="Slow SMA")
    buys = df[df["trade"] > 0]
    sells = df[df["trade"] < 0]
    plt.scatter(buys["date"], buys["close"], marker="^", label="Buy")
    plt.scatter(sells["date"], sells["close"], marker="v", label="Sell")
    plt.title("Price with SMA Crossover Signals")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "price_signals.png"))
    plt.close()

    # Equity curve
    plt.figure(figsize=(10,6))
    plt.plot(df["date"], df["equity_curve"], label="Equity Curve")
    plt.title("Strategy Equity Curve")
    plt.xlabel("Date"); plt.ylabel("Equity (normalized)"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "equity_curve.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Optional path to OHLCV CSV")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--fee_bps", type=float, default=5.0)
    parser.add_argument("--initial_cash", type=float, default=10000.0)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.csv is None:
        print("No CSV provided. Generating synthetic data...")
        df = generate_synthetic_prices()
    else:
        df = pd.read_csv(args.csv)

    df, metrics, trade_log = backtest_sma(df, args.fast, args.slow, args.fee_bps, args.initial_cash)
    save_outputs(df, metrics, trade_log, args.out_dir)

    print("\n=== Backtest Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nOutputs saved in: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
