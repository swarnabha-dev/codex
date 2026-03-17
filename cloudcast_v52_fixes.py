"""CloudCast v5.2 patch set.

Drop-in fixes for:
1) Metric inconsistency between validation (pixel-wise) and test (event-wise).
2) Flat rainfall predictions due to probability-weighted mean collapse.
3) Stronger temporal backbone with multi-layer bidirectional residual Dilated ConvLSTM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) CONSISTENT EVENT METRICS
# -----------------------------

def _event_scores_from_maps(
    prob_maps: np.ndarray,
    label_maps: np.ndarray,
    spatial_percentile: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse (B,H,W) maps -> event-level scalar arrays.

    Validation previously used flattened per-pixel CSI while test used one score per day,
    producing incomparable CSI values. This helper standardizes both to event-level.
    """
    if prob_maps.ndim != 3 or label_maps.ndim != 3:
        raise ValueError("Expected (B,H,W) tensors for event metric computation")

    pred_scores = np.percentile(prob_maps, spatial_percentile, axis=(1, 2)).astype(np.float64)
    true_events = (label_maps.max(axis=(1, 2)) >= 0.5).astype(np.float64)
    return pred_scores, true_events


def compute_event_metrics(
    prob_maps: np.ndarray,
    label_maps: np.ndarray,
    threshold: float,
    spatial_percentile: float = 95.0,
) -> dict:
    """Event-level metrics (CSI/POD/FAR/F1) consistent with test logic."""
    pred_scores, true_events = _event_scores_from_maps(prob_maps, label_maps, spatial_percentile)
    pred_events = (pred_scores >= threshold).astype(np.int32)
    true_events = true_events.astype(np.int32)

    hits = int(((pred_events == 1) & (true_events == 1)).sum())
    misses = int(((pred_events == 0) & (true_events == 1)).sum())
    false_alarms = int(((pred_events == 1) & (true_events == 0)).sum())

    csi = hits / max(hits + misses + false_alarms, 1)
    pod = hits / max(hits + misses, 1)
    far = false_alarms / max(hits + false_alarms, 1)
    f1 = 2 * hits / max(2 * hits + misses + false_alarms, 1)

    return {
        "CSI": csi,
        "POD": pod,
        "FAR": far,
        "F1": f1,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
    }


def calibrate_threshold_event_level(
    prob_maps: np.ndarray,
    label_maps: np.ndarray,
    spatial_percentile: float = 95.0,
) -> float:
    """Calibrate threshold on event-level CSI, not flattened pixels."""
    pred_scores, true_events = _event_scores_from_maps(prob_maps, label_maps, spatial_percentile)

    best_thr, best_csi = 0.5, -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        pred = (pred_scores >= thr).astype(np.int32)
        truth = true_events.astype(np.int32)
        h = ((pred == 1) & (truth == 1)).sum()
        m = ((pred == 0) & (truth == 1)).sum()
        fa = ((pred == 1) & (truth == 0)).sum()
        csi = h / max(h + m + fa, 1)
        if csi > best_csi:
            best_csi, best_thr = float(csi), float(thr)

    return best_thr


# ------------------------------------
# 2) LESS-FLAT RAINFALL AGGREGATION
# ------------------------------------

def aggregate_rainfall_prediction(
    mean_prob_hw: np.ndarray,
    mean_rain_hw: np.ndarray,
    percentile: float = 98.0,
    min_prob_gate: float = 0.35,
) -> float:
    """Convert rain map to scalar without collapsing to a near-constant mean.

    Old approach: weighted mean by probability -> often flat ~20-30 mm.
    New approach:
      1) Keep only confident rain pixels.
      2) Return high percentile to preserve event intensity variability.
    """
    if mean_prob_hw.shape != mean_rain_hw.shape:
        raise ValueError("Probability and rainfall maps must have same shape")

    rain = np.maximum(mean_rain_hw.astype(np.float64), 0.0)
    mask = mean_prob_hw >= min_prob_gate

    if mask.any():
        candidate = rain[mask]
    else:
        # fallback: top-10% probability area
        flat_idx = np.argsort(mean_prob_hw.ravel())
        k = max(1, int(0.10 * flat_idx.size))
        keep = flat_idx[-k:]
        candidate = rain.ravel()[keep]

    return float(np.percentile(candidate, percentile))


class SpatiallyWeightedRainHuber(nn.Module):
    """Improved regression loss weighting heavy-rain pixels stronger."""

    def __init__(self, rain_log_std: float = 1.0, heavy_rain_mm: float = 50.0, heavy_gain: float = 3.0):
        super().__init__()
        self.rain_log_std = max(float(rain_log_std), 1e-3)
        self.heavy_rain_mm = heavy_rain_mm
        self.heavy_gain = heavy_gain

    def forward(self, rain_logit: torch.Tensor, rain_spatial_true: torch.Tensor, prob_map: torch.Tensor) -> torch.Tensor:
        rain_log_true = torch.log1p(rain_spatial_true.clamp(min=0.0))
        base = F.huber_loss(
            rain_logit / self.rain_log_std,
            rain_log_true / self.rain_log_std,
            delta=1.0,
            reduction="none",
        )

        # preserve gradient in rainy areas + amplify extremes
        rain_gate = ((rain_spatial_true > 1.0) | (prob_map.detach() > 0.1)).float()
        heavy = (rain_spatial_true >= self.heavy_rain_mm).float()
        w = rain_gate * (1.0 + self.heavy_gain * heavy)

        denom = w.sum().clamp(min=1.0)
        return (base * w).sum() / denom


# -------------------------------------------------
# 3) MULTI-LAYER BIDIRECTIONAL RESIDUAL DIL CONVLSTM
# -------------------------------------------------

class DilConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, dil: int = 1):
        super().__init__()
        self.hid_ch = hid_ch
        self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size=3, padding=dil, dilation=dil)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gates(torch.cat([x, h], dim=1))
        i, f, gg, o = torch.chunk(g, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        gg = torch.tanh(gg)
        c_new = f * c + i * gg
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class _UniDirectionalStack(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, dils: List[int], dropout: float):
        super().__init__()
        self.cells = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        self.drop = nn.ModuleList()

        ch = in_ch
        for d in dils:
            self.cells.append(DilConvLSTMCell(ch, hid_ch, d))
            self.res_proj.append(nn.Conv2d(ch, hid_ch, kernel_size=1) if ch != hid_ch else nn.Identity())
            self.drop.append(nn.Dropout2d(dropout))
            ch = hid_ch

    def forward(self, x_seq: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        # x_seq: (B,T,C,H,W) -> out: (B,T,HID,H,W)
        if reverse:
            x_seq = torch.flip(x_seq, dims=[1])

        B, T, _, H, W = x_seq.shape
        out_seq = x_seq

        for li, cell in enumerate(self.cells):
            h = torch.zeros(B, cell.hid_ch, H, W, device=x_seq.device, dtype=x_seq.dtype)
            c = torch.zeros_like(h)
            states = []
            for t in range(T):
                xt = out_seq[:, t]
                h, c = cell(xt, h, c)
                # residual per layer for stronger gradient flow
                h = self.drop[li](h + self.res_proj[li](xt))
                states.append(h)
            out_seq = torch.stack(states, dim=1)

        if reverse:
            out_seq = torch.flip(out_seq, dims=[1])

        return out_seq


class BiResidualDilConvLSTM(nn.Module):
    """Multi-layer + bidirectional + residual temporal encoder.

    Output channel count is 2*hid_ch because forward/backward streams are concatenated.
    """

    def __init__(self, in_ch: int, hid_ch: int, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        dils = [1, 2, 4, 8][: max(1, n_layers)]
        self.fwd = _UniDirectionalStack(in_ch, hid_ch, dils, dropout)
        self.bwd = _UniDirectionalStack(in_ch, hid_ch, dils, dropout)
        self.merge = nn.Sequential(
            nn.Conv2d(2 * hid_ch, 2 * hid_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * hid_ch, 2 * hid_ch, kernel_size=1),
        )

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.fwd(x_seq, reverse=False)  # (B,T,HID,H,W)
        b = self.bwd(x_seq, reverse=True)   # (B,T,HID,H,W)
        y = torch.cat([f, b], dim=2)

        # final state map + sequence for temporal attention
        y_last = y[:, -1]
        y_last = self.merge(y_last)
        return y_last, y


@dataclass
class ArchitectureWeaknessChecklist:
    """Concise audit list to guide integration into the full training script."""

    weakness: str
    fix: str


ARCHITECTURE_AUDIT: Iterable[ArchitectureWeaknessChecklist] = [
    ArchitectureWeaknessChecklist(
        weakness="Validation/test metric mismatch (pixel-level vs event-level)",
        fix="Use event-level CSI calibration and logging via `compute_event_metrics` in both loops.",
    ),
    ArchitectureWeaknessChecklist(
        weakness="Rainfall scalar collapses to near-constant weighted mean",
        fix="Use percentile-based scalar (`aggregate_rainfall_prediction`) and heavy-rain weighted loss.",
    ),
    ArchitectureWeaknessChecklist(
        weakness="Temporal encoder is unidirectional; limited context and weaker gradients",
        fix="Replace with `BiResidualDilConvLSTM` to add bidirectional context + residual learning.",
    ),
]
