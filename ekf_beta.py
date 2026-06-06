import numpy as np

def extended_kalman_filter_beta(returns, macro_series, Q=0.01, R=1.0, initial_beta=0.0, initial_P=1.0):
    """
    Estimate time-varying beta for a single macro variable using EKF.
    Model: y_t = beta_t * x_t + epsilon_t, with beta_t = beta_{t-1} + w_t.
    Returns the final beta estimate (last step).
    """
    n = len(returns)
    if n < 3:
        return 0.0
    beta = initial_beta
    P = initial_P
    for t in range(n):
        x = macro_series[t]
        y = returns[t]
        # Prediction step (beta stays same)
        beta_pred = beta
        P_pred = P + Q
        # Update step (Kalman gain)
        H = x  # observation matrix (scalar)
        S = H * P_pred * H + R
        if S <= 0:
            continue
        K = P_pred * H / S
        # Innovation
        innov = y - H * beta_pred
        # Update beta
        beta = beta_pred + K * innov
        P = (1 - K * H) * P_pred
    return float(beta)

def ekf_macro_beta_score(returns, macro_df, primary_macro, Q=0.01, R=1.0, initial_beta=0.0):
    """
    For a given ETF, compute the time-varying beta of its returns with respect to the primary macro.
    Returns the last estimated beta (score).
    """
    if len(returns) != len(macro_df):
        min_len = min(len(returns), len(macro_df))
        returns = returns[:min_len]
        macro_df = macro_df.iloc[:min_len]
    if len(returns) < 5:
        return 0.0
    macro_series = macro_df[primary_macro].values
    # Standardise macro for numerical stability
    macro_std = macro_series.std()
    if macro_std < 1e-8:
        return 0.0
    macro_series = (macro_series - macro_series.mean()) / macro_std
    beta = extended_kalman_filter_beta(returns, macro_series, Q, R, initial_beta)
    # beta is scaled; we can leave as is.
    return beta
