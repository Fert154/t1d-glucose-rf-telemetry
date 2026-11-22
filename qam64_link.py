import numpy as np
from scipy.signal import correlate

R_DEFAULT = 32
DAC_SCALE = 2 ** 13
LEVELS = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float)
NORM = np.sqrt(42.0)
N_META_SYM = 13

def golay_pair(n=16):
    if n == 1:
        return np.array([1.0]), np.array([1.0])
    a, b = golay_pair(n // 2)
    return np.concatenate([a, b]), np.concatenate([a, -b])

def preamble_symbols():
    a, b = golay_pair(16)
    g = np.concatenate([a, b])
    return g * (7 + 7j) / NORM

def idx_to_sym(idx):
    idx = np.asarray(idx, dtype=int)
    return (LEVELS[idx // 8] + 1j * LEVELS[idx % 8]) / NORM

def sym_to_idx(sym):
    z = np.asarray(sym) * NORM
    i = np.clip(np.round((z.real + 7) / 2), 0, 7).astype(int)
    q = np.clip(np.round((z.imag + 7) / 2), 0, 7).astype(int)
    return 8 * i + q

def ideal_points():
    return idx_to_sym(np.arange(64))

def _bits(value, width):
    return [(int(value) >> (width - 1 - k)) & 1 for k in range(width)]

def _int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v

def _f32_bits(value):
    return _bits(int(np.array(value, dtype=np.float32).view(np.uint32)), 32)

def _bits_f32(bits):
    return float(np.array(_int(bits), dtype=np.uint32).view(np.float32))

def encode_meta(x_min, x_max, n):
    bits = _f32_bits(x_min) + _f32_bits(x_max) + _bits(n, 12) + [0, 0]
    return np.array([_int(bits[6 * k:6 * k + 6]) for k in range(N_META_SYM)])

def decode_meta(meta_idx):
    bits = []
    for k in meta_idx:
        bits += _bits(k, 6)
    return _bits_f32(bits[:32]), _bits_f32(bits[32:64]), _int(bits[64:76])

def build_frame(x, R=R_DEFAULT, scale=DAC_SCALE):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if not 1 <= n <= 4095:
        raise ValueError("La senal debe tener entre 1 y 4095 muestras (12 bits de N).")
    if not np.all(np.isfinite(x)):
        raise ValueError("La senal contiene NaN o infinitos.")
    lo = float(np.float32(np.min(x)))
    hi = float(np.float32(np.max(x)))
    span = hi - lo if hi > lo else 1.0
    idx = np.clip(np.round((x - lo) / span * 63), 0, 63).astype(int)
    meta_idx = encode_meta(lo, hi, n)

    symbols = np.concatenate([preamble_symbols(), idx_to_sym(meta_idx), idx_to_sym(idx)])
    frame = (np.repeat(symbols, R) * scale).astype(np.complex64)
    info = dict(x=x, idx=idx, meta_idx=meta_idx, x_min=lo, x_max=hi, n=n, R=R,
                n_symbols=len(symbols), frame_len=len(frame),
                x_quantized=lo + idx / 63.0 * span)
    return frame, info

def _symbol_samples(y, start, n_sym, R):
    a, b = R // 4, R - R // 4
    seg = y[start:start + n_sym * R]
    if len(seg) < n_sym * R:
        raise ValueError("El buffer de RX no contiene una trama completa despues del preambulo.")
    return seg.reshape(n_sym, R)[:, a:b].mean(axis=1)

def _find_peak(c, guess, R):
    lo, hi = max(guess - R, 0), min(guess + R, len(c) - 1)
    return lo + int(np.argmax(c[lo:hi + 1]))

def _track_phase(sym, loop_gain=0.05):
    out = np.empty_like(sym)
    phi = 0.0
    for k, s in enumerate(sym):
        z = s * np.exp(-1j * phi)
        d = idx_to_sym(sym_to_idx(z))
        phi += loop_gain * np.angle(z * np.conj(d))
        out[k] = z
    return out

def decode_frame(rx, R=R_DEFAULT, track_phase=True):
    rx = np.asarray(rx, dtype=complex)
    pre = np.repeat(preamble_symbols(), R)
    Lp = len(pre)
    c = np.abs(correlate(rx, pre, mode="valid"))

    usable = max(len(c) - N_META_SYM * R, 1)
    start = int(np.flatnonzero(c[:usable] >= 0.9 * c[:usable].max())[0])
    start = _find_peak(c, start, R)

    h1 = np.vdot(pre, rx[start:start + Lp]) / np.vdot(pre, pre)

    meta_start = start + Lp
    meta_sym = _symbol_samples(rx / h1, meta_start, N_META_SYM, R)
    meta_idx = sym_to_idx(meta_sym)
    lo, hi, n = decode_meta(meta_idx)
    if n == 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        raise ValueError("Metadatos invalidos: revisa nivel de senal y sincronizacion.")
    frame_len = (len(preamble_symbols()) + N_META_SYM + n) * R

    w, cfo_ok = 0.0, False
    nxt = start + frame_len
    if nxt + R < len(c):
        nxt = _find_peak(c, nxt, R)
        h2 = np.vdot(pre, rx[nxt:nxt + Lp]) / np.vdot(pre, pre)
        w = np.angle(h2 / h1) / (nxt - start)
        cfo_ok = True
    nn = np.arange(len(rx)) - (start + Lp / 2)
    y = rx * np.exp(-1j * w * nn) / h1

    meta_sym = _symbol_samples(y, meta_start, N_META_SYM, R)
    data_start = meta_start + N_META_SYM * R
    data_sym = _symbol_samples(y, data_start, n, R)
    if track_phase:
        data_sym = _track_phase(data_sym)
    idx = sym_to_idx(data_sym)
    span = hi - lo if hi > lo else 1.0
    x_hat = lo + idx / 63.0 * span

    return dict(x_hat=x_hat, idx=idx, data_sym=data_sym, meta_sym=meta_sym,
                meta_idx=meta_idx, x_min=lo, x_max=hi, n=n, start=start,
                frame_len=frame_len, channel_gain=h1,
                cfo_hz_per_fs=w / (2 * np.pi), cfo_estimated=cfo_ok,
                corr=c, rx_corrected=y)

def link_metrics(info, dec):
    sym = dec["data_sym"]
    ref = idx_to_sym(sym_to_idx(sym))
    p_ref = np.mean(np.abs(ideal_points()) ** 2)
    evm = np.sqrt(np.mean(np.abs(sym - ref) ** 2) / p_ref)

    n = min(len(info["idx"]), len(dec["idx"]))
    tx_i, rx_i = info["idx"][:n], dec["idx"][:n]
    ser = float(np.mean(tx_i != rx_i))
    bit_err = sum(bin(int(a) ^ int(b)).count("1") for a, b in zip(tx_i, rx_i))
    ber = bit_err / (6 * n)

    x, xh = info["x"][:n], dec["x_hat"][:n]
    r = float(np.corrcoef(x, xh)[0, 1])
    nmse = float(np.sum((x - xh) ** 2) / np.sum(x ** 2) * 100)
    xq = info["x_quantized"][:n]
    return dict(evm_rms=float(evm), evm_pct=float(100 * evm), ser=ser, ber=ber,
                bit_errors=int(bit_err), n_bits=6 * n, pearson_r=r, nmse_pct=nmse,
                max_abs_err=float(np.max(np.abs(x - xh))),
                quant_step=float((info["x_max"] - info["x_min"]) / 63),
                r_quantization_only=float(np.corrcoef(x, xq)[0, 1]),
                meta_ok=bool(np.isclose(dec["x_min"], info["x_min"]) and
                             np.isclose(dec["x_max"], info["x_max"]) and dec["n"] == info["n"]))
