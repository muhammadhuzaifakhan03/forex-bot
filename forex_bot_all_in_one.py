"""
forex_bot_all_in_one.py
------------------------
Full ML forex/gold/BTC trading bot pipeline in a single file:
data -> features -> ML model -> risk management -> walk-forward backtest ->
self-improving retraining -> paper execution -> MT5/Exness bridge.

Run directly: python3 forex_bot_all_in_one.py

NOTE: data_feed's synthetic generator is a placeholder so this runs without
internet access. Swap get_historical_data() for a real feed before using
this for anything beyond pipeline testing.
"""
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ======================================================================
# From config.py
# ======================================================================
# Per-asset trading specs. pip_value / point_size differ a lot between an FX
# pair, Gold, and BTC -- getting these wrong silently breaks position sizing
# and cost modeling, so they're centralized here instead of hardcoded.
ASSET_SPECS = {
    "EURUSD": {"point_size": 0.0001, "typical_spread_points": 1.0, "contract_size": 100_000},
    "GBPUSD": {"point_size": 0.0001, "typical_spread_points": 1.2, "contract_size": 100_000},
    "USDJPY": {"point_size": 0.01,   "typical_spread_points": 1.0, "contract_size": 100_000},
    "XAUUSD": {"point_size": 0.01,   "typical_spread_points": 20.0, "contract_size": 100},     # Gold
    "BTCUSD": {"point_size": 1.0,    "typical_spread_points": 15.0, "contract_size": 1},        # BTC
}

TIMEFRAMES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440,
}


@dataclass
class TradingConfig:
    asset: str = "EURUSD"                 # any key in ASSET_SPECS
    timeframe: str = "M15"                # any key in TIMEFRAMES
    lot_size: float = 0.10                # in standard lots (1.0 = 100k units for FX)
    max_risk_per_trade: float = 0.01      # fraction of equity, used if auto-sizing
    use_fixed_lot_size: bool = True       # True = always trade `lot_size`; False = risk-based auto-sizing
    confidence_threshold: float = 0.45
    signal_horizon_bars: int = 15         # how many bars ahead the model predicts

    def __post_init__(self):
        if self.asset not in ASSET_SPECS:
            raise ValueError(f"Unknown asset '{self.asset}'. Add it to ASSET_SPECS in config.py first.")
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(f"Unknown timeframe '{self.timeframe}'. Choose from {list(TIMEFRAMES)}.")

    @property
    def spec(self):
        return ASSET_SPECS[self.asset]

    @property
    def bar_minutes(self):
        return TIMEFRAMES[self.timeframe]

    def units_from_lots(self, lots: float) -> float:
        """Converts a lot size into raw units, asset-aware (1 lot means different things per asset)."""
        return lots * self.spec["contract_size"]


# ======================================================================
# From data_feed.py
# ======================================================================
# Base price level and daily-vol scale per asset, so synthetic data is at
# least in the right ballpark (Gold ~2000s, BTC ~60000s, FX pairs ~1.0-1.5).
_ASSET_BASE = {
    "EURUSD": {"base_price": 1.1000, "vol_scale": 1.0},
    "GBPUSD": {"base_price": 1.2700, "vol_scale": 1.1},
    "USDJPY": {"base_price": 150.00, "vol_scale": 90.0},
    "XAUUSD": {"base_price": 2350.0, "vol_scale": 900.0},
    "BTCUSD": {"base_price": 62000.0, "vol_scale": 40000.0},
}


def generate_synthetic_fx_data(n_bars: int = 5000, seed: int = 42, pair: str = "EURUSD") -> pd.DataFrame:
    """
    Generates plausible-looking minute-bar price data using a mean-reverting
    random walk with volatility clustering (GARCH-like), scaled per asset so
    Gold/BTC/FX all land at realistic price levels. Statistical shape only --
    NOT a real substitute for actual market data.
    """
    rng = np.random.default_rng(seed)
    n = n_bars
    asset_cfg = _ASSET_BASE.get(pair, _ASSET_BASE["EURUSD"])
    long_run_mean = asset_cfg["base_price"]
    vol_scale = asset_cfg["vol_scale"] / _ASSET_BASE["EURUSD"]["vol_scale"]  # relative to EURUSD baseline

    # Volatility clustering via a simple GARCH(1,1)-like process
    omega, alpha, beta = 1e-7 * vol_scale**2, 0.08, 0.90
    variances = np.zeros(n)
    variances[0] = omega / (1 - alpha - beta)
    shocks = rng.standard_normal(n)
    returns = np.zeros(n)
    for t in range(1, n):
        variances[t] = omega + alpha * returns[t - 1] ** 2 + beta * variances[t - 1]
        returns[t] = shocks[t] * np.sqrt(variances[t])

    # Slight mean-reversion + tiny drift so it doesn't look like pure noise
    price = np.zeros(n)
    price[0] = long_run_mean
    theta = 0.002  # mean reversion strength
    for t in range(1, n):
        price[t] = price[t - 1] + theta * (long_run_mean - price[t - 1]) + returns[t]

    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="min")
    close = price
    noise_scale = 0.00015 * vol_scale
    high = close + np.abs(rng.normal(0, noise_scale, n))
    low = close - np.abs(rng.normal(0, noise_scale, n))
    open_ = close + rng.normal(0, noise_scale * 0.7, n)
    volume = rng.integers(50, 500, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    df.attrs["pair"] = pair
    return df


def get_historical_data(n_bars: int = 5000, pair: str = "EURUSD") -> pd.DataFrame:
    """Entry point used by the rest of the pipeline. Swap this for a real API call."""
    return generate_synthetic_fx_data(n_bars=n_bars, pair=pair)


def get_latest_bar(history: pd.DataFrame, seed_offset: int = 0) -> pd.Series:
    """
    Simulates a new incoming bar for live/paper trading demos.
    Replace with a real streaming feed subscription in production.
    """
    rng = np.random.default_rng(1000 + seed_offset)
    last_close = history["close"].iloc[-1]
    new_close = last_close + rng.normal(0, 0.0002)
    new_bar = pd.Series(
        {
            "open": last_close,
            "high": max(last_close, new_close) + abs(rng.normal(0, 0.0001)),
            "low": min(last_close, new_close) - abs(rng.normal(0, 0.0001)),
            "close": new_close,
            "volume": rng.integers(50, 500),
        },
        name=history.index[-1] + pd.Timedelta(minutes=1),
    )
    return new_bar


# ======================================================================
# From features.py
# ======================================================================
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given raw OHLCV, returns a DataFrame of engineered features (unlagged, causal only)."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # Returns over multiple horizons
    for lag in (1, 5, 15, 30):
        out[f"ret_{lag}"] = close.pct_change(lag)

    # Moving averages / trend
    for window in (5, 20, 50):
        out[f"sma_{window}"] = close.rolling(window).mean()
        out[f"dist_sma_{window}"] = (close - out[f"sma_{window}"]) / close

    # Volatility
    out["volatility_20"] = close.pct_change().rolling(20).std()
    out["high_low_range"] = (df["high"] - df["low"]) / close

    # Momentum
    out["rsi_14"] = _rsi(close, 14)
    out["momentum_10"] = close - close.shift(10)

    # Volume
    out["volume_z"] = (df["volume"] - df["volume"].rolling(50).mean()) / df["volume"].rolling(50).std()

    return out


def build_target(df: pd.DataFrame, horizon: int = 15, threshold: float = 0.0001) -> pd.Series:
    """
    Target: direction of the return `horizon` bars ahead.
    1 = up beyond threshold, -1 = down beyond threshold, 0 = flat/no-trade zone.
    The no-trade zone matters: forcing a prediction on noise-level moves is
    how naive bots overtrade and bleed on spread/slippage.
    """
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    target = pd.Series(0, index=df.index)
    target[future_ret > threshold] = 1
    target[future_ret < -threshold] = -1
    return target


def make_dataset(df: pd.DataFrame, horizon: int = 15, threshold: float = 0.0001):
    """Returns (X, y) aligned and with NaNs dropped."""
    X = build_features(df)
    y = build_target(df, horizon=horizon, threshold=threshold)
    combined = X.copy()
    combined["target"] = y
    combined = combined.dropna()
    return combined.drop(columns="target"), combined["target"]


# ======================================================================
# From model.py
# ======================================================================
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class SignalModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.trained_at = None
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        self.trained_at = datetime.now(timezone.utc).isoformat()
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X[self.feature_names])
        return self.clf.predict_proba(Xs)

    def predict_signal(self, X, confidence_threshold: float = 0.45):
        """
        Returns -1 / 0 / 1 per row. Only takes a directional position when
        the model's confidence clears `confidence_threshold` -- otherwise 0
        (no trade). This confidence gate is a big part of what keeps a live
        bot from trading on noise.
        """
        proba = self.predict_proba(X)
        classes = self.clf.classes_  # e.g. [-1, 0, 1]
        best_idx = np.argmax(proba, axis=1)
        best_class = classes[best_idx]
        best_conf = proba[np.arange(len(proba)), best_idx]
        signal = np.where(best_conf >= confidence_threshold, best_class, 0)
        return signal, best_conf

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SignalModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ======================================================================
# From risk.py
# ======================================================================
@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01      # fraction of equity risked per trade
    stop_loss_pct: float = 0.002          # 0.2% adverse move = stop out
    take_profit_pct: float = 0.004        # 2:1 reward:risk
    max_daily_loss_pct: float = 0.03      # halt trading for the day past this
    max_drawdown_pct: float = 0.15        # halt trading entirely past this
    max_open_positions: int = 3


class RiskManager:
    def __init__(self, config: RiskConfig, starting_equity: float):
        self.config = config
        self.equity = starting_equity
        self.peak_equity = starting_equity
        self.daily_start_equity = starting_equity
        self.open_positions = 0
        self.trading_halted = False
        self.halt_reason = None

    def position_size(self, equity: float, entry_price: float) -> float:
        """Fixed-fractional sizing: risk `max_risk_per_trade` of equity on the stop distance."""
        risk_amount = equity * self.config.max_risk_per_trade
        stop_distance = entry_price * self.config.stop_loss_pct
        units = risk_amount / stop_distance
        return units

    def check_pretrade(self) -> bool:
        """Call before placing any order. Returns False if trading should be blocked."""
        if self.trading_halted:
            return False
        if self.open_positions >= self.config.max_open_positions:
            return False
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown >= self.config.max_drawdown_pct:
            self.trading_halted = True
            self.halt_reason = f"max drawdown breached: {drawdown:.1%}"
            return False
        daily_loss = (self.daily_start_equity - self.equity) / self.daily_start_equity
        if daily_loss >= self.config.max_daily_loss_pct:
            self.halt_reason = f"daily loss limit breached: {daily_loss:.1%}"
            return False
        return True

    def update_equity(self, new_equity: float):
        self.equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)

    def new_day(self):
        self.daily_start_equity = self.equity
        if self.halt_reason and "daily loss" in self.halt_reason:
            self.halt_reason = None  # daily halt resets; drawdown halt does not


# ======================================================================
# From backtest.py
# ======================================================================
def run_walk_forward_backtest(
    df: pd.DataFrame,
    train_window: int = 2000,
    test_window: int = 200,
    spread_pips: float = 1.0,
    pip_value: float = 0.0001,
    slippage_pips: float = 0.2,
    starting_equity: float = 10_000.0,
    confidence_threshold: float = 0.45,
):
    """
    Walk-forward backtest: train on a rolling window, test on the next
    unseen chunk, then roll forward. This is what "improves day by day"
    actually looks like in a disciplined pipeline -- the model is
    periodically retrained on the freshest data and evaluated out-of-sample
    before ever touching live capital.
    """
    X_all, y_all = make_dataset(df)
    aligned_close = df["close"].reindex(X_all.index)

    equity = starting_equity
    risk_mgr = RiskManager(RiskConfig(), starting_equity)
    equity_curve = []
    trade_log = []
    fold_scores = []

    n = len(X_all)
    start = 0
    while start + train_window + test_window <= n:
        train_X = X_all.iloc[start : start + train_window]
        train_y = y_all.iloc[start : start + train_window]
        test_X = X_all.iloc[start + train_window : start + train_window + test_window]
        test_y = y_all.iloc[start + train_window : start + train_window + test_window]
        test_close = aligned_close.iloc[start + train_window : start + train_window + test_window]

        model = SignalModel().fit(train_X, train_y)
        signals, confidences = model.predict_signal(test_X, confidence_threshold)

        # Simple next-bar-return simulation with costs applied on entries
        cost_per_trade = (spread_pips + slippage_pips) * pip_value

        prev_signal = 0
        for i in range(len(signals) - 1):
            sig = signals[i]
            price_now = test_close.iloc[i]
            price_next = test_close.iloc[i + 1]

            if not risk_mgr.check_pretrade():
                sig = 0  # trading halted

            if sig != 0:
                raw_ret = sig * (price_next - price_now) / price_now
                traded_this_bar = sig != prev_signal
                cost = cost_per_trade / price_now if traded_this_bar else 0.0
                net_ret = raw_ret - cost
                pnl = equity * net_ret
                equity += pnl
                risk_mgr.update_equity(equity)
                trade_log.append(
                    {
                        "timestamp": test_close.index[i],
                        "signal": sig,
                        "confidence": confidences[i],
                        "pnl": pnl,
                        "equity": equity,
                    }
                )
            prev_signal = sig
            equity_curve.append({"timestamp": test_close.index[i], "equity": equity})

        # Accuracy on the no-flat-zone subset as a fold health metric
        directional_mask = test_y != 0
        if directional_mask.sum() > 0:
            acc = (pd.Series(signals, index=test_y.index)[directional_mask] == test_y[directional_mask]).mean()
            fold_scores.append(acc)

        start += test_window

    equity_df = pd.DataFrame(equity_curve).set_index("timestamp") if equity_curve else pd.DataFrame()
    trades_df = pd.DataFrame(trade_log)

    summary = summarize_performance(equity_df, trades_df, starting_equity, fold_scores)
    return equity_df, trades_df, summary


def summarize_performance(equity_df, trades_df, starting_equity, fold_scores):
    if equity_df.empty:
        return {"error": "no trades executed -- confidence threshold may be too high"}

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity - starting_equity) / starting_equity

    returns = equity_df["equity"].pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)) if returns.std() > 0 else 0.0

    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - running_max) / running_max
    max_dd = drawdown.min()

    win_rate = (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "final_equity": round(final_equity, 2),
        "sharpe_ratio_annualized": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "num_trades": len(trades_df),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_fold_directional_accuracy_pct": round(np.mean(fold_scores) * 100, 2) if fold_scores else None,
    }


# ======================================================================
# From retrain.py
# ======================================================================
MODEL_DIR = Path("models")
REGISTRY_PATH = MODEL_DIR / "registry.json"


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"champion": None, "history": []}


def _save_registry(registry: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, default=str))


def _eval_model(model: SignalModel, X_eval, y_eval, close_eval, confidence_threshold=0.45):
    signals, conf = model.predict_signal(X_eval, confidence_threshold)
    prev = 0
    equity = 10_000.0
    curve = []
    for i in range(len(signals) - 1):
        sig = signals[i]
        if sig != 0:
            ret = sig * (close_eval.iloc[i + 1] - close_eval.iloc[i]) / close_eval.iloc[i]
            cost = 0.00012 if sig != prev else 0.0  # ~1.2 pip round-trip proxy
            equity *= (1 + ret - cost)
        prev = sig
        curve.append({"timestamp": close_eval.index[i], "equity": equity})
    equity_df = pd.DataFrame(curve).set_index("timestamp")
    return summarize_performance(equity_df, pd.DataFrame(), 10_000.0, [])


def run_retrain_cycle(df: pd.DataFrame, min_improvement_pct: float = 2.0):
    """
    Trains a challenger on all available data, evaluates both champion and
    challenger on the same recent held-out slice, and promotes the
    challenger only if it beats the champion's total return by at least
    `min_improvement_pct` percentage points. Returns the decision + both
    scorecards for logging/monitoring.
    """
    X, y = make_dataset(df)
    close = df["close"].reindex(X.index)

    split = int(len(X) * 0.85)
    train_X, eval_X = X.iloc[:split], X.iloc[split:]
    train_y, eval_y = y.iloc[:split], y.iloc[split:]
    eval_close = close.iloc[split:]

    challenger = SignalModel().fit(train_X, train_y)
    challenger_score = _eval_model(challenger, eval_X, eval_y, eval_close)

    registry = _load_registry()
    champion_score = None
    promote = True

    if registry["champion"] is not None:
        champion_path = registry["champion"]["path"]
        champion = SignalModel.load(champion_path)
        champion_score = _eval_model(champion, eval_X, eval_y, eval_close)

        champ_return = champion_score.get("total_return_pct", -999)
        chall_return = challenger_score.get("total_return_pct", -999)
        promote = chall_return >= champ_return + min_improvement_pct

    decision_time = datetime.now(timezone.utc).isoformat()
    result = {
        "timestamp": decision_time,
        "challenger_score": challenger_score,
        "champion_score": champion_score,
        "promoted": promote,
    }

    if promote:
        version_tag = decision_time.replace(":", "-")
        new_path = str(MODEL_DIR / f"model_{version_tag}.pkl")
        challenger.save(new_path)
        registry["champion"] = {"path": new_path, "promoted_at": decision_time, "score": challenger_score}
        registry["history"].append(result)
        _save_registry(registry)

    return result


# ======================================================================
# From execution.py
# ======================================================================
@dataclass
class Position:
    pair: str
    direction: int  # 1 = long, -1 = short
    units: float
    entry_price: float
    opened_at: str


class PaperExecutionClient:
    """Simulated broker connection. Same interface a real one should expose."""

    def __init__(self, starting_balance: float = 10_000.0):
        self.balance = starting_balance
        self.positions: dict[str, Position] = {}
        self.fill_log = []

    def place_order(self, pair: str, direction: int, units: float, price: float):
        if direction == 0:
            return None
        pos = Position(
            pair=pair,
            direction=direction,
            units=units,
            entry_price=price,
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        self.positions[pair] = pos
        self.fill_log.append({"pair": pair, "direction": direction, "units": units, "price": price})
        return pos

    def close_position(self, pair: str, price: float):
        pos = self.positions.pop(pair, None)
        if pos is None:
            return 0.0
        pnl = pos.direction * (price - pos.entry_price) * pos.units
        self.balance += pnl
        self.fill_log.append({"pair": pair, "action": "close", "price": price, "pnl": pnl})
        return pnl

    def get_open_position(self, pair: str):
        return self.positions.get(pair)


# ======================================================================
# From mt5_bridge.py
# ======================================================================
class MT5Bridge:
    def __init__(self, mt5_files_dir: str, symbol: str = "EURUSD"):
        self.dir = Path(mt5_files_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.signal_path = self.dir / "bot_signal.json"
        self.report_path = self.dir / "bot_execution_report.json"

    def send_signal(self, direction: int, lot_size: float, stop_loss_price: float = None,
                     take_profit_price: float = None, confidence: float = None):
        """
        direction: 1 = buy, -1 = sell, 0 = close/flat
        Writes atomically (write to temp file then rename) so the EA never
        reads a half-written file.
        """
        payload = {
            "symbol": self.symbol,
            "direction": direction,
            "lot_size": lot_size,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "confidence": confidence,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path = self.signal_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload))
        tmp_path.replace(self.signal_path)
        return payload

    def read_execution_report(self, timeout_seconds: float = 10.0, poll_interval: float = 0.5):
        """Waits for the EA to write back a fill confirmation, returns it or None on timeout."""
        waited = 0.0
        while waited < timeout_seconds:
            if self.report_path.exists():
                try:
                    data = json.loads(self.report_path.read_text())
                    self.report_path.unlink(missing_ok=True)
                    return data
                except (json.JSONDecodeError, OSError):
                    pass  # EA may still be mid-write; retry
            time.sleep(poll_interval)
            waited += poll_interval
        return None


def generate_ea_file(output_path: str = "bridge_ea.mq5"):
    """
    Writes out the companion MQL5 Expert Advisor source. Compile this in
    MetaEditor (comes with MT5) and attach it to any chart for the symbol
    you're trading. Works with Exness or any standard MT5 broker -- nothing
    Exness-specific is needed since Exness is a plain MT5 broker.
    """
    ea_code = r"""
//+------------------------------------------------------------------+
//| bridge_ea.mq5                                                    |
//| Reads signals written by the Python bot (mt5_bridge.py) from a   |
//| JSON file in MQL5/Files, and places/closes orders accordingly.   |
//| Works on any standard MT5 broker, including Exness.              |
//+------------------------------------------------------------------+
#property strict

input string SignalFileName   = "bot_signal.json";
input string ReportFileName   = "bot_execution_report.json";
input int    PollSeconds      = 2;

int OnInit()
{
   EventSetTimer(PollSeconds);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   string content = ReadFile(SignalFileName);
   if(content == "") return;

   int direction = ExtractInt(content, "\"direction\":");
   double lots   = ExtractDouble(content, "\"lot_size\":");
   double sl     = ExtractDouble(content, "\"stop_loss_price\":");
   double tp     = ExtractDouble(content, "\"take_profit_price\":");
   string symbol = ExtractString(content, "\"symbol\":");
   if(symbol == "") symbol = _Symbol;

   ExecuteSignal(symbol, direction, lots, sl, tp);
}

void ExecuteSignal(string symbol, int direction, double lots, double sl, double tp)
{
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   bool hasPosition = PositionSelect(symbol);

   if(direction == 0)
   {
      if(hasPosition) ClosePosition(symbol);
      return;
   }

   if(hasPosition)
   {
      long posType = PositionGetInteger(POSITION_TYPE);
      bool isLong = (posType == POSITION_TYPE_BUY);
      if((direction == 1 && !isLong) || (direction == -1 && isLong))
         ClosePosition(symbol);
      else
         return;
   }

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = symbol;
   request.volume    = lots;
   request.type      = (direction == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price     = (direction == 1) ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                                         : SymbolInfoDouble(symbol, SYMBOL_BID);
   request.sl        = sl;
   request.tp        = tp;
   request.deviation = 10;
   request.magic     = 990011;

   bool sent = OrderSend(request, result);
   WriteReport(symbol, sent, result);
}

void ClosePosition(string symbol)
{
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   long posType = PositionGetInteger(POSITION_TYPE);
   double volume = PositionGetDouble(POSITION_VOLUME);

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = symbol;
   request.volume    = volume;
   request.type      = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price     = (posType == POSITION_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_BID)
                                                       : SymbolInfoDouble(symbol, SYMBOL_ASK);
   request.deviation = 10;
   request.magic     = 990011;
   request.position  = PositionGetInteger(POSITION_TICKET);

   OrderSend(request, result);
}

void WriteReport(string symbol, bool sent, MqlTradeResult &result)
{
   int handle = FileOpen(ReportFileName, FILE_WRITE|FILE_TXT|FILE_COMMON);
   if(handle == INVALID_HANDLE) return;
   string json = StringFormat(
      "{\"symbol\":\"%s\",\"sent\":%s,\"retcode\":%d,\"price\":%.5f,\"balance\":%.2f}",
      symbol, sent ? "true" : "false", result.retcode, result.price,
      AccountInfoDouble(ACCOUNT_BALANCE));
   FileWriteString(handle, json);
   FileClose(handle);
}

string ReadFile(string name)
{
   int handle = FileOpen(name, FILE_READ|FILE_TXT|FILE_COMMON);
   if(handle == INVALID_HANDLE) return "";
   string content = "";
   while(!FileIsEnding(handle))
      content += FileReadString(handle);
   FileClose(handle);
   return content;
}

int ExtractInt(string src, string key)
{
   int pos = StringFind(src, key);
   if(pos < 0) return 0;
   pos += StringLen(key);
   return (int)StringToInteger(StringSubstr(src, pos, 5));
}

double ExtractDouble(string src, string key)
{
   int pos = StringFind(src, key);
   if(pos < 0) return 0.0;
   pos += StringLen(key);
   return StringToDouble(StringSubstr(src, pos, 15));
}

string ExtractString(string src, string key)
{
   int pos = StringFind(src, key);
   if(pos < 0) return "";
   pos = StringFind(src, "\"", pos + StringLen(key)) + 1;
   int endPos = StringFind(src, "\"", pos);
   if(pos < 0 || endPos < 0) return "";
   return StringSubstr(src, pos, endPos - pos);
}
"""
    Path(output_path).write_text(ea_code)
    return output_path


# ======================================================================
# From main.py
# ======================================================================
def main(config: TradingConfig = None):
    config = config or TradingConfig(asset="EURUSD", timeframe="M15", lot_size=0.10)

    print("=" * 60)
    print(f"Config: asset={config.asset}  timeframe={config.timeframe}  lot_size={config.lot_size}")
    print("=" * 60)

    print("\n1. Fetching data (synthetic placeholder -- swap for real feed)")
    df = get_historical_data(n_bars=6000, pair=config.asset)
    print(f"Loaded {len(df)} bars for {config.asset}")

    print("\n2. Running walk-forward backtest")
    equity_df, trades_df, summary = run_walk_forward_backtest(
        df, confidence_threshold=config.confidence_threshold
    )
    print(json.dumps(summary, indent=2))

    print("\n3. Running a self-improvement retrain cycle")
    result = run_retrain_cycle(df)
    print(f"Promoted new champion: {result['promoted']}")
    print("Challenger score:", json.dumps(result["challenger_score"], indent=2))
    if result["champion_score"]:
        print("Previous champion score:", json.dumps(result["champion_score"], indent=2))

    return equity_df, trades_df, summary, result


if __name__ == "__main__":
    # 1. Run the full data -> model -> backtest -> retrain demo.
    #    Try any of: TradingConfig(asset="XAUUSD", timeframe="M15", lot_size=0.05)
    #                TradingConfig(asset="BTCUSD", timeframe="H1", lot_size=0.01)
    main(TradingConfig(asset="EURUSD", timeframe="M15", lot_size=0.10))

    # 2. Generate the companion MQL5 Expert Advisor file for MT5/Exness.
    ea_path = generate_ea_file()
    print(f"\nGenerated {ea_path} -- copy into your MT5 MQL5/Experts folder and compile in MetaEditor.")
