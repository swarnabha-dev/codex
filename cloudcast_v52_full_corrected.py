import os, gc, json, math, time, pickle, hashlib, shutil, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CONFIG = {
    "model_name": "CloudCast",
    "version": "v5.2-corrected",
    "atmos_dir": "Atmosphere",
    "rainfall_csv": "RAINFALL.csv",
    "orography_path": None,
    "output_dir": "output",
    "checkpoint_dir": "output/checkpoints",
    "cache_dir": "output/cache",
    "lat_min": 17.0, "lat_max": 29.0,
    "lon_min": 80.0, "lon_max": 93.0,
    "grid_step": 0.25,
    "pressure_levels": [200, 300, 500, 700, 850, 925, 1000],
    "variables": ["ciwc", "clwc", "crwc", "q", "r", "t", "u", "v", "w", "z"],
    "cloudburst_threshold_mm": 100.0,
    "lookback_steps": 3,
    "train_frac": 0.70,
    "val_frac": 0.15,
    "epochs": 10,
    "batch_size": 4,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 15,
    "overfit_gap_thr": 0.30,
    "hidden_dim": 64,
    "lstm_layers": 3,
    "dropout": 0.25,
    "mc_passes": 8,
    "focal_gamma": 2.0,
    "focal_alpha": 0.75,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "tversky_weight": 0.4,
    "regression_weight": 0.7,
    "aux_loss_weight": 0.10,
    "hard_neg_ratio": 10,
    "spatial_percentile": 95,
    "rain_scalar_percentile": 98,
    "rain_prob_gate": 0.35,
    "reg_output_ceil": 6.2166,
    "norm_stats_path": "output/norm_stats.json",
    "phys_stats_path": "output/phys_stats.json",
    "pixel_weight_path": "output/pixel_weight.json",
    "rain_stats_path": "output/rain_stats.json",
    "decision_threshold": None,
    "seed": 42,
    "num_workers": 2,
    "pin_memory": True,
    "use_amp": True,
    "resume": True,
    "log_every": 5,
    "diagnostic_only": False,
}

_DP = {200: 10000, 300: 15000, 500: 20000, 700: 17500, 850: 11250, 925: 7500, 1000: 5075}
N_CH = 33


def seed_everything(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def make_dirs(cfg):
    for k in ["output_dir", "checkpoint_dir", "cache_dir"]:
        Path(cfg[k]).mkdir(parents=True, exist_ok=True)


def _to_py(v):
    if isinstance(v, dict): return {k: _to_py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_to_py(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if hasattr(v, "item"): return v.item()
    return v


def jdump(obj, path): json.dump(_to_py(obj), open(path, "w"), indent=2)

def jload(path): return json.load(open(path))


def resolve_atmos_files(cfg):
    d = cfg.get("atmos_dir")
    if d and Path(d).is_dir():
        f = sorted(Path(d).glob("*.parquet"))
        if not f: raise FileNotFoundError(f"No .parquet in {d}")
        return [str(x) for x in f]
    raise FileNotFoundError("No atmos_dir.")


def config_hash(cfg):
    keys = ["atmos_dir", "rainfall_csv", "pressure_levels", "variables", "lat_min", "lat_max", "lon_min", "lon_max", "grid_step", "lookback_steps", "cloudburst_threshold_mm"]
    blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True) + "_v52corrected"
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def detect_level_col(df):
    for c in ["level", "isobaricInhPa", "pressure_level", "plev"]:
        if c in df.columns: return c
    return None


def safe_theta_e(T_K, p_hPa, q_kgkg):
    Lv = 2.5e6; Cp = 1004.0; Rd = 287.0
    q = max(min(float(q_kgkg), 0.05), 0.0)
    T_K = max(float(T_K), 200.0)
    theta = T_K * (1000.0 / max(float(p_hPa), 1.0)) ** (Rd / Cp)
    exponent = min(Lv * q / (Cp * T_K), 0.5)
    return theta * math.exp(exponent)


def dewpoint_C(T_K, rh_pct):
    T_C = T_K - 273.15
    rh = max(min(rh_pct, 100.0), 0.1)
    ln_rh = math.log(rh / 100.0)
    a, b = 17.67, 243.5
    gamma = a * T_C / (b + T_C) + ln_rh
    return b * gamma / (a - gamma)


def pressure_weighted_pw(q_by_lv):
    return sum(q_by_lv.get(lv, 0.0) * _DP.get(lv, 10000) for lv in q_by_lv) / 9.81


def compute_convergence(u, v, q, dx, dy):
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    div = (du_dx + dv_dy).astype(np.float32)
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    vort = (dv_dx - du_dy).astype(np.float32)
    dq_dx = np.gradient(q, dx, axis=1)
    dq_dy = np.gradient(q, dy, axis=0)
    mfc = -(q * div + u * dq_dx + v * dq_dy).astype(np.float32)
    return div, vort, mfc


def grid_spacing(lat_deg, dlat, dlon):
    R = 6371000.0
    return R * np.radians(dlon) * np.cos(np.radians(lat_deg)), R * np.radians(dlat)


def get_date_index(cfg, cache_root):
    pkl = Path(cache_root) / f"date_idx_{config_hash(cfg)}.pkl"
    if pkl.exists():
        return pickle.load(open(pkl, "rb"))
    d2f = {}
    for p in resolve_atmos_files(cfg):
        t = pd.read_parquet(p, columns=["time"])
        t["time"] = pd.to_datetime(t["time"])
        for d in t["time"].dt.normalize().unique():
            d2f.setdefault(d, []).append(p)
    pickle.dump(d2f, open(pkl, "wb"))
    return d2f


class CloudCastDataset(Dataset):
    def __init__(self, cfg, split, split_info, norm_mu=0.0, norm_std=1.0):
        self.cfg = cfg
        self.split = split
        self.cache_dir = Path(cfg["cache_dir"]) / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.norm_mu = norm_mu
        self.norm_std = norm_std
        self.levels = cfg["pressure_levels"]
        self.lb = cfg["lookback_steps"]
        self.lats = np.arange(cfg["lat_min"], cfg["lat_max"] + 1e-9, cfg["grid_step"])
        self.lons = np.arange(cfg["lon_min"], cfg["lon_max"] + 1e-9, cfg["grid_step"])
        self.H = len(self.lats)
        self.W = len(self.lons)
        lat_c = (cfg["lat_min"] + cfg["lat_max"]) / 2
        self.dx, self.dy = grid_spacing(lat_c, cfg["grid_step"], cfg["grid_step"])
        self.orog = torch.zeros(1, self.H, self.W)
        self.index = self._build_or_load_cache(split_info)

    def _build_or_load_cache(self, split_info):
        idx_f = self.cache_dir / "index.json"
        hash_f = self.cache_dir / f".hash_{config_hash(self.cfg)}"
        if idx_f.exists() and hash_f.exists():
            return jload(idx_f)
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        rain = pd.read_csv(self.cfg["rainfall_csv"])
        rain.columns = rain.columns.str.strip().str.lower()
        if "rainfall" in rain.columns and "rainfall_mm" not in rain.columns:
            rain = rain.rename(columns={"rainfall": "rainfall_mm"})
        rain["time"] = pd.to_datetime(rain["time"])
        rain["date"] = rain["time"].dt.normalize()
        rain = rain[rain.latitude.between(self.cfg["lat_min"], self.cfg["lat_max"]) & rain.longitude.between(self.cfg["lon_min"], self.cfg["lon_max"])].dropna(subset=["rainfall_mm"])

        d2f = get_date_index(self.cfg, self.cache_dir.parent)
        all_dates = sorted(d2f.keys())
        d2i = {d: i for i, d in enumerate(all_dates)}
        index = []

        for ts in split_info["timestamps"]:
            ts_date = pd.Timestamp(ts).normalize()
            ts_idx = d2i.get(ts_date)
            if ts_idx is None: continue
            lb_dates = []
            ok = True
            for step in range(self.lb - 1, -1, -1):
                if ts_idx - step < 0:
                    ok = False
                    break
                lb_dates.append(all_dates[ts_idx - step])
            if not ok: continue
            needed = set()
            for d in lb_dates: needed.update(d2f.get(d, []))
            needed.update(d2f.get(ts_date, []))
            frames = []
            for fp in needed:
                part = pd.read_parquet(fp)
                part["time"] = pd.to_datetime(part["time"])
                lc = detect_level_col(part)
                if lc and lc != "level": part = part.rename(columns={lc: "level"})
                part["date"] = part["time"].dt.normalize()
                frames.append(part[part["date"].isin(set(lb_dates) | {ts_date})])
            if not frames: continue
            chunk = pd.concat(frames, ignore_index=True)

            x = self._build_input(chunk, lb_dates)
            label = self._build_label(rain, ts_date, self.cfg["cloudburst_threshold_mm"])
            rain_max = self._rain_max(rain, ts_date)
            rain_spatial = self._rain_spatial(rain, ts_date)
            phys = self._physics(chunk, ts_date)

            fname = self.cache_dir / f"{ts_date.strftime('%Y%m%d')}.pt"
            torch.save({"x": x, "label": label, "rain_max": rain_max, "rain_spatial": rain_spatial, "phys": phys, "ts": str(ts_date)}, fname)
            index.append({"ts": str(ts_date), "path": str(fname), "has_event": int(label.max().item() >= 0.5), "rain_max": float(rain_max), "n_phys": 4})

        jdump(index, idx_f); hash_f.touch()

        if self.split == "train" and len(index):
            all_x = []
            phys_a = []
            px_pos = px_neg = 0
            rain_vals = []
            for it in index:
                d = torch.load(it["path"], map_location="cpu", weights_only=False)
                x = d["x"].numpy().ravel()
                x = x[np.isfinite(x)]
                all_x.append(x)
                phys_a.append(d["phys"].numpy())
                lab = d["label"].numpy()
                px_pos += int((lab >= 0.5).sum()); px_neg += int((lab < 0.5).sum())
                if d["rain_max"] > 0: rain_vals.append(d["rain_max"])
            xcat = np.concatenate(all_x) if all_x else np.array([0.0])
            jdump({"mu": float(xcat.mean()), "std": float(xcat.std() + 1e-6)}, self.cfg["norm_stats_path"])
            pa = np.stack(phys_a) if phys_a else np.zeros((1, 4), np.float32)
            jdump({"mu": pa.mean(0).tolist(), "std": (pa.std(0) + 1e-6).tolist()}, self.cfg["phys_stats_path"])
            jdump({"pixel_pos_weight": float(px_neg / max(px_pos, 1))}, self.cfg["pixel_weight_path"])
            rv = np.array(rain_vals) if rain_vals else np.array([1.0])
            jdump({"mean": float(rv.mean()), "std": float(rv.std() + 1e-6), "max": float(rv.max())}, self.cfg["rain_stats_path"])

        return index

    def _agg(self, rows, fi, var):
        g_mn = np.zeros(self.H * self.W, np.float32)
        g_mx = np.zeros(self.H * self.W, np.float32)
        g_sd = np.zeros(self.H * self.W, np.float32)
        if var not in rows.columns or len(rows) == 0: return g_mn, g_mx, g_sd
        vals = rows[var].values.astype(np.float32)
        ok = np.isfinite(vals)
        if not ok.any(): return g_mn, g_mx, g_sd
        ag = pd.DataFrame({"i": fi[ok], "v": vals[ok]}).groupby("i")["v"].agg(["mean", "max", "std"])
        g_mn[ag.index.values] = ag["mean"].values
        g_mx[ag.index.values] = ag["max"].values
        g_sd[ag.index.values] = ag["std"].fillna(0).values
        return g_mn, g_mx, g_sd

    def _build_input(self, atmos_df, lb_dates):
        steps = []
        for date in lb_dates:
            day = atmos_df[atmos_df["date"] == date]
            hi = np.abs(day["latitude"].values[:, None] - self.lats[None, :]).argmin(axis=1)
            wi = np.abs(day["longitude"].values[:, None] - self.lons[None, :]).argmin(axis=1)
            fi = hi * self.W + wi
            level_grids = []
            for lv in self.levels:
                if "level" in day.columns:
                    mask = day["level"].values == lv
                    rows = day[mask]; lfi = fi[mask]
                else:
                    rows = day; lfi = fi
                ch = np.zeros((self.H * self.W, N_CH), np.float32)
                for gi, var in enumerate(["q", "r", "t", "w", "z"]):
                    mn, mx, sd = self._agg(rows, lfi, var)
                    ch[:, gi * 3:(gi + 1) * 3] = np.stack([mn, mx, sd], axis=1)
                u_mn, u_mx, u_sd = self._agg(rows, lfi, "u")
                v_mn, v_mx, v_sd = self._agg(rows, lfi, "v")
                ch[:, 15] = np.sqrt(u_mn ** 2 + v_mn ** 2)
                ch[:, 16] = np.sqrt(u_mx ** 2 + v_mx ** 2)
                ch[:, 17] = np.sqrt(u_sd ** 2 + v_sd ** 2)
                ci, _, _ = self._agg(rows, lfi, "ciwc")
                cl, _, _ = self._agg(rows, lfi, "clwc")
                cr, cr_mx, cr_sd = self._agg(rows, lfi, "crwc")
                tot = ci + cl + cr
                safe_tot = np.where(tot > 1e-12, tot, 1e-12)
                ch[:, 18:21] = np.stack([tot, np.maximum(np.maximum(self._agg(rows, lfi, "ciwc")[1], self._agg(rows, lfi, "clwc")[1]), cr_mx), np.sqrt(self._agg(rows, lfi, "ciwc")[2] ** 2 + self._agg(rows, lfi, "clwc")[2] ** 2 + cr_sd ** 2)], axis=1)
                ch[:, 21] = np.clip(ci / safe_tot, 0, 1)
                ch[:, 22] = ch[:, 21]
                ch[:, 24] = np.clip(cr / safe_tot, 0, 1)
                ch[:, 25] = ch[:, 24]
                if lv == 850:
                    div, vort, mfc = compute_convergence(u_mn.reshape(self.H, self.W), v_mn.reshape(self.H, self.W), self._agg(rows, lfi, "q")[0].reshape(self.H, self.W), self.dx, self.dy)
                    ch[:, 30] = div.ravel(); ch[:, 31] = vort.ravel(); ch[:, 32] = mfc.ravel()
                level_grids.append(ch.reshape(self.H, self.W, N_CH))
            steps.append(np.stack(level_grids, axis=0))
        return torch.tensor(np.stack(steps, axis=0), dtype=torch.float32)

    def _build_label(self, rain, date, thr):
        day = rain[rain["date"] == date]
        lab = np.zeros((self.H, self.W), np.float32)
        grp = day.groupby(["latitude", "longitude"])["rainfall_mm"].max() if not day.empty else []
        for (lat, lon), val in grp.items():
            hi = np.argmin(np.abs(self.lats - lat)); wi = np.argmin(np.abs(self.lons - lon))
            lab[hi, wi] = 1.0 if val >= thr else 0.0
        return torch.tensor(lab)

    def _rain_max(self, rain, date):
        day = rain[rain["date"] == date]
        if day.empty: return 0.0
        v = day["rainfall_mm"].max()
        return float(v) if np.isfinite(v) else 0.0

    def _rain_spatial(self, rain, date):
        day = rain[rain["date"] == date]
        grid = np.zeros((self.H, self.W), np.float32)
        if not day.empty:
            grp = day.groupby(["latitude", "longitude"])["rainfall_mm"].max()
            for (lat, lon), val in grp.items():
                hi = np.argmin(np.abs(self.lats - lat)); wi = np.argmin(np.abs(self.lons - lon))
                grid[hi, wi] = max(float(val), 0.0)
        return torch.tensor(grid, dtype=torch.float32)

    def _physics(self, atmos_df, date):
        day = atmos_df[atmos_df["date"] == date]
        def lvm(lv, var):
            sub = day[day["level"] == lv] if "level" in day.columns else day
            if var not in sub.columns or sub.empty: return None
            v = sub[var].values.astype(np.float64)
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else None
        t850 = lvm(850, "t"); t500 = lvm(500, "t"); t700 = lvm(700, "t")
        r850 = lvm(850, "r"); r700 = lvm(700, "r"); q850 = lvm(850, "q")
        cape_p = safe_theta_e(t850, 850, q850) - safe_theta_e(t500, 500, lvm(500, "q") or 1e-5) if (t850 and t500 and q850) else 0.0
        kidx = ((t850 - 273.15) + dewpoint_C(t850, r850) - (t500 - 273.15) - (t700 - 273.15) + dewpoint_C(t700, r700)) if (t850 and t500 and t700 and r850 and r700) else 0.0
        q_lv = {lv: lvm(lv, "q") for lv in self.levels if lvm(lv, "q") is not None}
        pw = pressure_weighted_pw(q_lv)
        return torch.tensor([float(cape_p), float(kidx), float(pw), 0.0], dtype=torch.float32)

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        d = torch.load(self.index[idx]["path"], map_location="cpu", weights_only=False)
        x = torch.nan_to_num((d["x"] - self.norm_mu) / max(self.norm_std, 1e-6), 0.0)
        ts = d["ts"]
        doy = pd.Timestamp(ts).day_of_year
        season = torch.tensor([math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365)], dtype=torch.float32)
        return {"x": x, "label": d["label"], "phys": d["phys"], "rain_max": torch.tensor([d.get("rain_max", 0.0)], dtype=torch.float32), "rain_spatial": d.get("rain_spatial", torch.zeros(self.H, self.W)), "season": season, "orog": self.orog, "ts": ts}


def build_temporal_split(cfg):
    all_dates = set()
    for p in resolve_atmos_files(cfg):
        df = pd.read_parquet(p, columns=["time"])
        df["time"] = pd.to_datetime(df["time"])
        all_dates.update(df["time"].dt.normalize().unique().tolist())
    all_dates = sorted(all_dates)
    N = len(all_dates)
    nt = int(N * cfg["train_frac"]); nv = int(N * cfg["val_frac"])
    return {"train": {"timestamps": all_dates[:nt]}, "val": {"timestamps": all_dates[nt:nt + nv]}, "test": {"timestamps": all_dates[nt + nv:]}}


class PressurePositionalEmbedding(nn.Module):
    def __init__(self, n_levels, embed_dim):
        super().__init__(); self.emb = nn.Embedding(n_levels, embed_dim)
    def forward(self, L, dev): return self.emb(torch.arange(L, device=dev)).view(1, L, 1, 1, -1)


class CausalPE(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        pe = torch.zeros(T, C)
        pos = torch.arange(T).float().unsqueeze(1)
        half = C // 2
        div = torch.exp(torch.arange(0, half).float() * (-math.log(10000) / max(half, 1)))
        pe[:, :half] = torch.sin(pos * div)
        pe[:, half:half * 2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:x.shape[1]].view(1, x.shape[1], x.shape[2], 1, 1)


class VariableProj(nn.Module):
    def __init__(self, in_ch, out_ch, dr=0.1):
        super().__init__(); self.net = nn.Sequential(nn.Linear(in_ch, out_ch), nn.LayerNorm(out_ch), nn.GELU(), nn.Dropout(dr))
    def forward(self, x): return self.net(x)


class CrossLevelAttn(nn.Module):
    def __init__(self, n_levels, E, n_heads=4, dr=0.1):
        super().__init__()
        n_heads = max(1, min(n_heads, E // 8 if E >= 8 else 1))
        while E % n_heads != 0: n_heads -= 1
        self.attn = nn.MultiheadAttention(E, n_heads, dropout=dr, batch_first=True)
        self.norm = nn.LayerNorm(E); self.drop = nn.Dropout(dr)
        self.pe = PressurePositionalEmbedding(n_levels, E)
    def forward(self, x):
        B, L, H, W, C = x.shape
        x = x + self.pe(L, x.device).expand(B, -1, H, W, -1)
        xr = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, L, C)
        out, _ = self.attn(xr, xr, xr)
        out = self.norm(xr + self.drop(out))
        return out.reshape(B, H, W, L, C).permute(0, 3, 1, 2, 4)


class LevelMixer(nn.Module):
    def __init__(self, n_levels, n_raw, E, C, dr=0.1):
        super().__init__()
        self.vproj = VariableProj(n_raw, E, dr)
        self.attn = CrossLevelAttn(n_levels, E, n_heads=4, dr=dr)
        self.proj = nn.Sequential(nn.Linear(n_levels * E, C * 2), nn.GELU(), nn.Dropout(dr), nn.Linear(C * 2, C))
    def forward(self, x):
        B, T, L, H, W, V = x.shape
        out = []
        for t in range(T):
            xt = self.proj(self.attn(self.vproj(x[:, t])).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1))
            out.append(xt.permute(0, 3, 1, 2))
        return torch.stack(out, dim=1)


class DilConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, dil=1):
        super().__init__(); self.hid_ch = hid_ch; self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, 3, padding=dil, dilation=dil)
    def forward(self, x, h, c):
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class _UniDirResidualDilConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, dils, dr=0.2):
        super().__init__()
        self.cells = nn.ModuleList()
        self.res = nn.ModuleList()
        self.drop = nn.ModuleList()
        ch = in_ch
        for d in dils:
            self.cells.append(DilConvLSTMCell(ch, hid_ch, d))
            self.res.append(nn.Conv2d(ch, hid_ch, 1) if ch != hid_ch else nn.Identity())
            self.drop.append(nn.Dropout2d(dr))
            ch = hid_ch

    def forward(self, x_seq, reverse=False):
        if reverse: x_seq = torch.flip(x_seq, dims=[1])
        B, T, C, H, W = x_seq.shape
        out_seq = x_seq
        for li, cell in enumerate(self.cells):
            h = torch.zeros(B, cell.hid_ch, H, W, device=x_seq.device, dtype=x_seq.dtype)
            c = torch.zeros_like(h)
            states = []
            for t in range(T):
                xt = out_seq[:, t]
                h, c = cell(xt, h, c)
                h = self.drop[li](h + self.res[li](xt))
                states.append(h)
            out_seq = torch.stack(states, dim=1)
        if reverse: out_seq = torch.flip(out_seq, dims=[1])
        return out_seq


class BiResidualDilConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, n_layers=3, dr=0.2):
        super().__init__()
        dils = [1, 2, 4, 8][:max(1, n_layers)]
        self.fwd = _UniDirResidualDilConvLSTM(in_ch, hid_ch, dils, dr)
        self.bwd = _UniDirResidualDilConvLSTM(in_ch, hid_ch, dils, dr)
        self.merge = nn.Sequential(nn.Conv2d(2 * hid_ch, 2 * hid_ch, 3, padding=1), nn.GELU(), nn.Conv2d(2 * hid_ch, 2 * hid_ch, 1))
    def forward(self, x):
        f = self.fwd(x, reverse=False)
        b = self.bwd(x, reverse=True)
        y = torch.cat([f, b], dim=2)
        return self.merge(y[:, -1]), y


class TemporalAttn(nn.Module):
    def __init__(self, C, T, n_heads=4, dr=0.1):
        super().__init__()
        n_heads = max(1, min(n_heads, C // 8 if C >= 8 else 1))
        while C % n_heads != 0: n_heads -= 1
        self.attn = nn.MultiheadAttention(C, n_heads, dropout=dr, batch_first=True)
        self.norm = nn.LayerNorm(C)
        self.pe = CausalPE(T, C)
    def forward(self, seq):
        x = self.pe(seq)
        B, T, C, H, W = x.shape
        xr = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        out, _ = self.attn(xr, xr, xr)
        out = self.norm(xr + out)
        return out.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)


class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        g = torch.sigmoid(self.conv(torch.cat([x.mean(1, True), x.max(1).values.unsqueeze(1)], 1)))
        return x * g


class UNetEnc(nn.Module):
    def __init__(self, in_ch, C, dr=0.1):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, C, 3, padding=1), nn.BatchNorm2d(C), nn.GELU())
        self.p1 = nn.Conv2d(C, C * 2, 3, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.Conv2d(C * 2, C * 2, 3, padding=1), nn.BatchNorm2d(C * 2), nn.GELU(), nn.Dropout2d(dr))
        self.p2 = nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1)
        self.e3 = nn.Sequential(nn.Conv2d(C * 4, C * 4, 3, padding=1), nn.BatchNorm2d(C * 4), nn.GELU(), nn.Dropout2d(dr))
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p1(e1)); e3 = self.e3(self.p2(e2)); return e3, [e1, e2]


class UNetDec(nn.Module):
    def __init__(self, C, dr=0.1):
        super().__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2 = nn.Sequential(nn.Conv2d(C * 4 + C * 2, C * 2, 3, padding=1), nn.BatchNorm2d(C * 2), nn.GELU(), nn.Dropout2d(dr))
        self.sa2 = SpatialAttn()
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1 = nn.Sequential(nn.Conv2d(C * 2 + C, C, 3, padding=1), nn.BatchNorm2d(C), nn.GELU(), nn.Dropout2d(dr))
        self.sa1 = SpatialAttn()
    def forward(self, bot, skips):
        e1, e2 = skips
        d = self.up2(bot)
        if d.shape[-2:] != e2.shape[-2:]: d = F.pad(d, [0, e2.shape[-1] - d.shape[-1], 0, e2.shape[-2] - d.shape[-2]])
        d = self.sa2(self.d2(torch.cat([d, e2], 1)))
        d = self.up1(d)
        if d.shape[-2:] != e1.shape[-2:]: d = F.pad(d, [0, e1.shape[-1] - d.shape[-1], 0, e1.shape[-2] - d.shape[-2]])
        return self.sa1(self.d1(torch.cat([d, e1], 1)))


class CloudCast(nn.Module):
    def __init__(self, cfg, n_levels, n_vars_raw):
        super().__init__()
        C = cfg["hidden_dim"]; E = C // 2; dr = cfg["dropout"]
        self.mixer = LevelMixer(n_levels, n_vars_raw, E, C, dr)
        self.orog_cv = nn.Conv2d(1, C // 4, 1)
        self.lstm = BiResidualDilConvLSTM(C + C // 4, C, n_layers=cfg["lstm_layers"], dr=dr)
        self.tattn = TemporalAttn(C * 2, cfg["lookback_steps"], 4, dr)
        self.uenc = UNetEnc(C * 2, C, dr)
        self.udec = UNetDec(C, dr)
        self.cls_head = nn.Sequential(nn.Conv2d(C, C // 2, 3, padding=1), nn.GELU(), nn.Conv2d(C // 2, 1, 1), nn.Sigmoid())
        self.reg_head = nn.Sequential(nn.Conv2d(C, C // 2, 3, padding=1), nn.GELU(), nn.Conv2d(C // 2, 1, 1))
        self.phys_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C * 2, 64), nn.GELU(), nn.Dropout(dr), nn.Linear(64, 4))

    def forward(self, x, season, orog, enable_dropout=False):
        if enable_dropout:
            for m in self.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d)): m.train()
        B, T, L, H, W, V = x.shape
        feat = self.mixer(x)
        feat = torch.cat([feat, self.orog_cv(orog).unsqueeze(1).expand(B, T, -1, H, W)], dim=2)
        h_final, seq = self.lstm(feat)
        attended = self.tattn(seq)
        feat_map = attended.max(dim=1).values
        phys = self.phys_head(h_final)
        bot, sk = self.uenc(feat_map)
        shared = self.udec(bot, sk)
        return self.cls_head(shared).squeeze(1), self.reg_head(shared).squeeze(1), phys


class HNMFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=1.0, hnm_ratio=10):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.pos_weight = pos_weight; self.hnm_ratio = hnm_ratio
    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        bce = -(self.pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        p_t = torch.where(target == 1, pred, 1 - pred)
        a_t = torch.where(target == 1, torch.full_like(pred, self.alpha), torch.full_like(pred, 1 - self.alpha))
        focal = a_t * (1 - p_t) ** self.gamma * bce
        out = []
        for b in range(pred.shape[0]):
            pos = focal[b][target[b] >= 0.5]; neg = focal[b][target[b] < 0.5]
            if pos.numel() > 0:
                n_hard = min(pos.numel() * self.hnm_ratio, neg.numel())
                out.append(torch.cat([pos, neg.topk(n_hard).values if n_hard > 0 else neg[:0]]).mean())
            else:
                n_s = max(int(neg.numel() * 0.01), 1)
                out.append(neg[torch.randperm(neg.numel(), device=neg.device)[:n_s]].mean())
        return torch.stack(out).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__(); self.alpha = alpha; self.beta = beta; self.smooth = smooth
    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        B = pred.shape[0]; p = pred.reshape(B, -1); t = target.reshape(B, -1)
        tp = (p * t).sum(1); fp = (p * (1 - t)).sum(1); fn = ((1 - p) * t).sum(1)
        return (1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)).mean()


class CloudCastLoss(nn.Module):
    def __init__(self, cfg, pixel_pw, phys_mu, phys_std, rain_log_std):
        super().__init__()
        self.focal = HNMFocalLoss(cfg["focal_alpha"], cfg["focal_gamma"], pixel_pw, cfg["hard_neg_ratio"])
        self.tversky = TverskyLoss(cfg["tversky_alpha"], cfg["tversky_beta"])
        self.tv_w = cfg["tversky_weight"]
        self.reg_w = cfg["regression_weight"]
        self.aux_w = cfg["aux_loss_weight"]
        self.rain_log_std = max(rain_log_std, 1e-3)
        self.register_buffer("phys_mu", phys_mu.float())
        self.register_buffer("phys_std", phys_std.float())

    def forward(self, prob_map, rain_logit, pred_phys, label_map, rain_max_true, rain_spatial_true, phys_targets):
        fl = self.focal(prob_map, label_map)
        tv = self.tversky(prob_map, label_map)
        rain_log_true = torch.log1p(rain_spatial_true.clamp(min=0.0))
        gate = ((prob_map.detach() > 0.1) | (rain_spatial_true > 1.0)).float()
        heavy = (rain_spatial_true >= 50.0).float()
        w = gate * (1.0 + 3.0 * heavy)
        reg = (F.huber_loss(rain_logit / self.rain_log_std, rain_log_true / self.rain_log_std, delta=1.0, reduction="none") * w).sum() / w.sum().clamp(min=1.0)
        aux = F.mse_loss(pred_phys, torch.nan_to_num((phys_targets - self.phys_mu) / (self.phys_std + 1e-6), 0.0))
        total = fl + self.tv_w * tv + self.reg_w * reg + self.aux_w * aux
        return total, fl.item(), tv.item(), reg.item(), aux.item()


def compute_metrics(preds, labels, threshold=0.5):
    from sklearn.metrics import average_precision_score, roc_auc_score
    pb = (preds >= threshold).astype(int); lb = (labels >= 0.5).astype(int)
    h = int(((pb == 1) & (lb == 1)).sum()); m = int(((pb == 0) & (lb == 1)).sum()); fa = int(((pb == 1) & (lb == 0)).sum())
    try:
        auc_pr = average_precision_score(lb.ravel(), preds.ravel())
        auc_roc = roc_auc_score(lb.ravel(), preds.ravel()) if lb.sum() > 0 else 0.0
    except Exception:
        auc_pr = auc_roc = 0.0
    return {"CSI": h / max(h + m + fa, 1), "POD": h / max(h + m, 1), "FAR": fa / max(h + fa, 1), "F1": 2 * h / max(2 * h + m + fa, 1), "AUC_PR": auc_pr, "AUC_ROC": auc_roc, "hits": h, "misses": m, "false_alarms": fa}


def compute_event_metrics_from_maps(prob_maps, label_maps, threshold, percentile=95):
    event_scores = np.percentile(prob_maps, percentile, axis=(1, 2))
    event_labels = (label_maps.max(axis=(1, 2)) >= 0.5).astype(float)
    return compute_metrics(event_scores, event_labels, threshold)


def regression_metrics(pred_mm, true_mm):
    v = np.isfinite(pred_mm) & np.isfinite(true_mm)
    if not v.any(): return {"MAE": 0.0, "RMSE": 0.0, "corr": 0.0}
    p, t = pred_mm[v], true_mm[v]
    return {"MAE": float(np.abs(p - t).mean()), "RMSE": float(np.sqrt(((p - t) ** 2).mean())), "corr": float(np.corrcoef(p, t)[0, 1]) if len(p) > 2 else 0.0}


def calibrate_threshold(model, val_dl, device, cfg):
    model.eval(); all_pm, all_lb = [], []
    with torch.no_grad():
        for batch in val_dl:
            pm, _, _ = model(batch["x"].to(device), batch["season"].to(device), batch["orog"].to(device))
            all_pm.append(pm.cpu().numpy())
            all_lb.append(batch["label"].numpy())
    prob_maps = np.concatenate(all_pm, axis=0)
    label_maps = np.concatenate(all_lb, axis=0)
    best_thr, best_csi = 0.5, -1
    for thr in np.arange(0.05, 0.95, 0.025):
        csi = compute_event_metrics_from_maps(prob_maps, label_maps, thr, cfg["spatial_percentile"])["CSI"]
        if csi > best_csi: best_csi, best_thr = csi, thr
    log(f"Calibrated EVENT threshold={best_thr:.3f} CSI={best_csi:.4f}")
    return float(best_thr)


def save_ckpt(state, path): torch.save(state, path)

def load_ckpt(model, opt, sched, cfg):
    ckpts = sorted(Path(cfg["checkpoint_dir"]).glob("epoch_*.pt"))
    if not ckpts or not cfg["resume"]: return 0, float("inf"), []
    ck = torch.load(ckpts[-1], map_location=get_device(), weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
    if sched and "scheduler" in ck: sched.load_state_dict(ck["scheduler"])
    return ck["epoch"] + 1, ck["best_val_loss"], ck.get("history", [])


def run_epoch(model, loader, criterion, opt, scaler, device, cfg, mode, threshold=0.5):
    is_tr = mode == "train"
    model.train() if is_tr else model.eval()
    n = 0; tot = fl = tv = rl = al = 0.0
    map_probs, map_labels = [], []
    rain_pred, rain_true = [], []
    dev = device.type
    ceil = cfg.get("reg_output_ceil", math.log1p(500))
    ctx = torch.enable_grad() if is_tr else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            x = batch["x"].to(device); lb = batch["label"].to(device)
            ph = batch["phys"].to(device); s = batch["season"].to(device)
            og = batch["orog"].to(device); rm = batch["rain_max"].to(device)
            rs = batch["rain_spatial"].to(device)
            with autocast(dev, enabled=cfg["use_amp"] and dev == "cuda"):
                pm, rain_logit, pp = model(x, s, og)
                loss, fl_v, tv_v, rl_v, al_v = criterion(pm, rain_logit, pp, lb, rm, rs, ph)
            if is_tr:
                opt.zero_grad(set_to_none=True)
                if cfg["use_amp"] and dev == "cuda":
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); opt.step()
            tot += loss.item(); fl += fl_v; tv += tv_v; rl += rl_v; al += al_v; n += 1
            map_probs.append(pm.detach().cpu().numpy())
            map_labels.append(lb.detach().cpu().numpy())
            rp = torch.expm1(rain_logit.clamp(0, ceil)).detach().cpu().numpy().reshape(rain_logit.shape[0], -1).max(axis=1)
            rain_pred.append(rp); rain_true.append(rm.detach().cpu().numpy().ravel())
            if is_tr and (i + 1) % cfg["log_every"] == 0:
                log(f"[{mode}] {i+1}/{len(loader)} loss={loss.item():.4f}")
    prob_maps = np.concatenate(map_probs, axis=0)
    label_maps = np.concatenate(map_labels, axis=0)
    m_evt = compute_event_metrics_from_maps(prob_maps, label_maps, threshold, cfg["spatial_percentile"])
    m_px = compute_metrics(prob_maps.ravel(), label_maps.ravel(), 0.5)
    m = {**m_evt, "AUC_PR_px": m_px["AUC_PR"], "AUC_ROC_px": m_px["AUC_ROC"], "pred_std": float(prob_maps.std()), "loss": tot / max(n, 1), "focal": fl / max(n, 1), "tversky": tv / max(n, 1), "reg": rl / max(n, 1), "aux": al / max(n, 1)}
    m.update(regression_metrics(np.concatenate(rain_pred), np.concatenate(rain_true)))
    return m


def aggregate_rain_scalar(prob_hw, rain_hw, cfg):
    mask = prob_hw >= cfg["rain_prob_gate"]
    if mask.any(): cand = rain_hw[mask]
    else:
        idx = np.argsort(prob_hw.ravel())
        k = max(1, int(0.1 * idx.size))
        cand = rain_hw.ravel()[idx[-k:]]
    return float(np.percentile(cand, cfg["rain_scalar_percentile"]))


def train(cfg):
    seed_everything(cfg["seed"]); device = get_device(); make_dirs(cfg)
    split = build_temporal_split(cfg)
    CloudCastDataset(cfg, "train", split["train"], 0.0, 1.0)
    mu_s = jload(cfg["norm_stats_path"]); mu, std = mu_s["mu"], mu_s["std"]
    ph_s = jload(cfg["phys_stats_path"])
    phys_mu = torch.tensor(ph_s["mu"], dtype=torch.float32)
    phys_std = torch.tensor(ph_s["std"], dtype=torch.float32)
    pixel_pw = jload(cfg["pixel_weight_path"])["pixel_pos_weight"]
    rain_log_std = float(np.log1p(jload(cfg["rain_stats_path"]).get("std", 10.0)))

    train_ds = CloudCastDataset(cfg, "train", split["train"], mu, std)
    val_ds = CloudCastDataset(cfg, "val", split["val"], mu, std)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])

    sample = train_ds[0]
    _, L, H, W, V = sample["x"].shape
    model = CloudCast(cfg, L, V).to(device)
    criterion = CloudCastLoss(cfg, pixel_pw, phys_mu, phys_std, rain_log_std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    scaler = GradScaler(enabled=cfg["use_amp"] and device.type == "cuda")

    start, best_vl, history = load_ckpt(model, opt, sched, cfg)
    threshold = 0.5
    patience = 0
    for epoch in range(start, cfg["epochs"]):
        tr = run_epoch(model, train_dl, criterion, opt, scaler, device, cfg, "train", threshold)
        va = run_epoch(model, val_dl, criterion, opt, scaler, device, cfg, "val", threshold)
        threshold = calibrate_threshold(model, val_dl, device, cfg)
        sched.step()
        log(f"train loss={tr['loss']:.4f} CSI_evt={tr['CSI']:.3f} AUC_PR_px={tr['AUC_PR_px']:.3f} MAE={tr['MAE']:.1f}")
        log(f"val   loss={va['loss']:.4f} CSI_evt={va['CSI']:.3f} AUC_PR_px={va['AUC_PR_px']:.3f} MAE={va['MAE']:.1f} thr={threshold:.3f}")
        history.append({"epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": va["loss"], "train_CSI": tr["CSI"], "val_CSI": va["CSI"], "train_MAE": tr["MAE"], "val_MAE": va["MAE"], "val_pred_std": va["pred_std"]})
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "best_val_loss": best_vl, "history": history, "cfg": cfg, "norm_mu": mu, "norm_std": std}
        if va["loss"] < best_vl:
            best_vl = va["loss"]; patience = 0
            save_ckpt(state, Path(cfg["checkpoint_dir"]) / "best.pt")
        else:
            patience += 1
        save_ckpt(state, Path(cfg["checkpoint_dir"]) / f"epoch_{epoch:03d}.pt")
        if patience >= cfg["patience"]: break

    cfg["decision_threshold"] = float(threshold)
    jdump({"threshold": threshold, "norm_mu": mu, "norm_std": std}, Path(cfg["output_dir"]) / "calibrated_threshold.json")
    return mu, std, float(threshold)


def test(cfg, mu=None, std=None, threshold=None):
    device = get_device(); out_dir = Path(cfg["output_dir"])
    cal = jload(out_dir / "calibrated_threshold.json") if (out_dir / "calibrated_threshold.json").exists() else {}
    mu = mu if mu is not None else cal.get("norm_mu", 0.0)
    std = std if std is not None else cal.get("norm_std", 1.0)
    threshold = threshold if threshold is not None else cal.get("threshold", 0.5)
    ceil = cfg.get("reg_output_ceil", math.log1p(500))

    split = build_temporal_split(cfg)
    test_ds = CloudCastDataset(cfg, "test", split["test"], mu, std)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    sample = test_ds[0]
    _, L, H, W, V = sample["x"].shape
    model = CloudCast(cfg, L, V).to(device)
    ckpt = Path(cfg["checkpoint_dir"]) / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval()

    pct = cfg["spatial_percentile"]
    results = []; preds_l = []; labels_l = []; rain_pred_l = []; rain_true_l = []
    for batch in test_dl:
        x = batch["x"].to(device); lb = batch["label"].to(device)
        s = batch["season"].to(device); og = batch["orog"].to(device)
        rm_true = float(batch["rain_max"].item()); ts = batch["ts"][0]
        mc_prob, mc_rain = [], []
        for _ in range(cfg["mc_passes"]):
            with torch.no_grad(): pm, rain_logit, _ = model(x, s, og, enable_dropout=True)
            mc_prob.append(pm.cpu().numpy()); mc_rain.append(torch.expm1(rain_logit.clamp(0, ceil)).cpu().numpy())
        mean_prob = np.stack(mc_prob).mean(0)[0]
        std_prob = np.stack(mc_prob).std(0)[0]
        mean_rain = np.stack(mc_rain).mean(0)[0]
        prob_scalar = float(np.percentile(mean_prob, pct))
        rain_scalar = aggregate_rain_scalar(mean_prob, mean_rain, cfg)
        true_ev = int(lb.cpu().numpy().max() >= 0.5)
        results.append({"timestamp": ts, "pred_prob_p95": round(prob_scalar, 4), "pred_unc": round(float(std_prob.max()), 4), "pred_binary": int(prob_scalar >= threshold), "true_event": true_ev, "pred_rainfall_mm": round(max(rain_scalar, 0.0), 2), "true_rainfall_mm": round(rm_true, 2), "threshold_used": round(float(threshold), 3)})
        preds_l.append(prob_scalar); labels_l.append(float(true_ev)); rain_pred_l.append(max(rain_scalar, 0.0)); rain_true_l.append(rm_true)

    metrics = compute_metrics(np.array(preds_l), np.array(labels_l), threshold)
    reg_m = regression_metrics(np.array(rain_pred_l), np.array(rain_true_l))
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "test_predictions.csv", index=False)
    jdump({**metrics, **{f"reg_{k}": v for k, v in reg_m.items()}}, out_dir / "test_metrics.json")
    plot_timeseries(df, metrics, reg_m, out_dir, cfg)
    plot_rainfall_trend(df, reg_m, out_dir, cfg)


def plot_timeseries(df, metrics, reg_m, out_dir, cfg):
    fig = plt.figure(figsize=(18, 12)); gs = gridspec.GridSpec(4, 1, hspace=0.50, figure=fig); x = range(len(df))
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(x, np.clip(df["pred_prob_p95"] - df["pred_unc"], 0, 1), np.clip(df["pred_prob_p95"] + df["pred_unc"], 0, 1), alpha=0.2, color="#1f77b4")
    ax1.plot(x, df["pred_prob_p95"], "#1f77b4", lw=1.5, label="P(cloudburst) p95")
    ax1.step(x, df["true_event"], where="mid", color="#d62728", lw=1.5, ls="--", label="GT event")
    thr = df["threshold_used"].iloc[0]; ax1.axhline(thr, color="gray", lw=0.8, ls=":", label=f"Thr={thr:.3f}")
    ax1.set_ylabel("Probability"); ax1.legend(fontsize=8); ax1.set_ylim(-0.05, 1.15)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x, df["true_rainfall_mm"], color="#d62728", lw=2.0, marker="o", ms=5, label="Ground truth (mm)")
    ax2.plot(x, df["pred_rainfall_mm"], color="#1f77b4", lw=1.5, marker="s", ms=4, label="Predicted (mm)", linestyle="--")
    if (df["pred_rainfall_mm"].max() > 2 * max(df["true_rainfall_mm"].max(), 1e-6)) or (df["true_rainfall_mm"].max() > 2 * max(df["pred_rainfall_mm"].max(), 1e-6)):
        ax2.set_yscale("log")
    ax2.set_ylabel("Rainfall (mm)"); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2]); ax3.bar(x, df["pred_unc"], color="#9467bd", alpha=0.7); ax3.set_ylabel("Uncertainty (std)")
    ax4 = fig.add_subplot(gs[3]); ax4.plot(x, np.array(df["pred_rainfall_mm"]) - np.array(df["true_rainfall_mm"]), color="#555555"); ax4.axhline(0, color="black", lw=1)

    step = max(1, len(df) // 20)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(list(range(0, len(df), step)))
        ax.set_xticklabels([df["timestamp"].iloc[i] for i in range(0, len(df), step)], rotation=45, ha="right", fontsize=7)
    fig.savefig(out_dir / "test_timeseries.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_rainfall_trend(df, reg_m, out_dir, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True); x = range(len(df)); ts = [str(df["timestamp"].iloc[i])[:10] for i in range(len(df))]
    axes[0].plot(x, df["true_rainfall_mm"], color="#d62728", lw=2.5, marker="o", ms=6, label="Ground truth")
    axes[0].plot(x, df["pred_rainfall_mm"], color="#1f77b4", lw=2.0, marker="s", ms=5, linestyle="--", label="Predicted")
    axes[0].legend(fontsize=9); axes[0].set_ylabel("Rainfall (mm)"); axes[0].grid(alpha=0.25)
    err = np.array(df["pred_rainfall_mm"]) - np.array(df["true_rainfall_mm"])
    axes[1].plot(x, err, color="#555555", lw=1.5, marker="D", ms=4); axes[1].axhline(0, color="black", lw=1.0)
    step = max(1, len(df) // 15)
    axes[1].set_xticks(list(range(0, len(df), step))); axes[1].set_xticklabels([ts[i] for i in range(0, len(df), step)], rotation=35, ha="right", fontsize=9)
    plt.tight_layout(); fig.savefig(out_dir / "rainfall_trend.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    make_dirs(CONFIG)
    if CONFIG["diagnostic_only"]: return
    jdump({k: v for k, v in CONFIG.items() if not callable(v)}, Path(CONFIG["output_dir"]) / "config.json")
    mu, std, threshold = train(CONFIG)
    test(CONFIG, mu, std, threshold)


if __name__ == "__main__":
    main()
