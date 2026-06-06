# Extended Kalman Filter for Macro Beta Tracking

Estimates time‑varying sensitivity (beta) of ETF returns to a macro variable (e.g., VIX) using an extended Kalman filter. The beta is modelled as a random walk. The per‑ETF score is the current beta estimate – positive means the ETF benefits from macro increases.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Uses primary macro (e.g., VIX) and optionally other macro variables (not yet in the filter but can be extended)
- EKF: state = beta, observation = returns * macro
- Score = final beta (time‑varying sensitivity)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-ekf-macro-beta-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- Positive beta → ETF tends to rise when the macro variable rises (e.g., VIX up → higher returns).
- Negative beta → inverse relationship.
- High absolute beta → high sensitivity, useful for hedging or directional bets.

## Requirements

See `requirements.txt`.
