
import numpy as np
import pandas as pd



def _normalize(sig: np.ndarray) -> np.ndarray:
    """We normalize to zero mean and unit variance when std > 0."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std > 0:
        sig = sig / std
    return sig






def estimate_global_lag(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
):
    """
    We estimate one global lag using sign-invariant cross-correlation.
    We return lag in seconds, lag in frames, and correlation curve.
    """
    s1 = _normalize(s_rppg)
    s2 = _normalize(s_ppg)

    T = len(s1)
    if T < 2:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0])

    corr = np.correlate(s1, s2, mode="full")
    lags = np.arange(-T + 1, T)

    max_lag_samples = int(min(max_lag_seconds / dt, T - 1))
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)

    lags_win = lags[mask]
    corr_win = corr[mask]

    best_idx = int(np.argmax(np.abs(corr_win)))
    best_samples = int(lags_win[best_idx])

    lag_sec = float(best_samples * dt)
    fps_est = 1.0 / dt
    lag_frames = lag_sec * fps_est

    return lag_sec, lag_frames, lags_win, corr_win




def estimate_local_lag_curve(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    local_win_seconds: float = 4.0,
    max_lag_seconds: float = 2.0,
):
    """
    We estimate a local lag for each frame using a sliding cross-correlation.

    For each center index, we take a small window around it, compute
    sign-invariant cross-correlation, and store the lag (in samples).

    We return an array lag_samples(t) of length T, and lag_frames(t).
    """
    s1 = _normalize(s_rppg)
    s2 = _normalize(s_ppg)

    T = len(s1)
    if T < 4:
        return np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64)

    fps_est = 1.0 / dt
    win_samples = int(round(local_win_seconds * fps_est))
    if win_samples < 5:
        win_samples = 5
    if win_samples % 2 == 0:
        win_samples += 1
    half = win_samples // 2

    max_lag_samples = int(min(max_lag_seconds * fps_est, half))

    lag_samples_curve = np.full(T, np.nan, dtype=np.float64)

    for c in range(half, T - half):
        seg1 = s1[c - half : c + half + 1]
        seg2 = s2[c - half : c + half + 1]

        seg1 = _normalize(seg1)
        seg2 = _normalize(seg2)

        corr = np.correlate(seg1, seg2, mode="full")
        lags = np.arange(-len(seg1) + 1, len(seg1))

        mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
        lags_loc = lags[mask]
        corr_loc = corr[mask]

        best_idx = int(np.argmax(np.abs(corr_loc)))
        best_lag = int(lags_loc[best_idx])
        lag_samples_curve[c] = best_lag

    # We fill NaNs at edges using nearest valid value
    valid = np.where(~np.isnan(lag_samples_curve))[0]
    if len(valid) == 0:
        lag_samples_curve[:] = 0.0
    else:
        first = valid[0]
        last = valid[-1]
        lag_samples_curve[:first] = lag_samples_curve[first]
        lag_samples_curve[last + 1 :] = lag_samples_curve[last]

    lag_frames_curve = lag_samples_curve.astype(np.float64)
    return lag_samples_curve, lag_frames_curve

