import sys
import numpy as np
from qam64_link import build_frame, decode_frame, link_metrics, R_DEFAULT

FS = 61.44e6
rng = np.random.default_rng(1)

def canal(frame, snr_db, cfo_hz=200.0, fase=1.1, ganancia=0.37, retardo=None, n_rx=131072, iq_eps=0.02):
    reps = int(np.ceil(n_rx * 2 / len(frame))) + 1
    tx = np.tile(frame, reps)
    if retardo is None:
        retardo = int(rng.integers(0, len(frame)))
    seg = tx[retardo:retardo + n_rx].astype(complex)
    t = np.arange(n_rx) / FS
    y = ganancia * np.exp(1j * (fase + 2 * np.pi * cfo_hz * t)) * seg
    y = y + iq_eps * np.conj(y)
    p = np.mean(np.abs(y) ** 2)
    ruido = np.sqrt(p / 10 ** (snr_db / 10) / 2) * (rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx))
    return y + ruido

if len(sys.argv) > 1:
    x = np.loadtxt(sys.argv[1], delimiter=",", usecols=0)
else:
    t = np.arange(1012)
    x = 150 + 80 * np.sin(2 * np.pi * t / 300) + 40 * np.sin(2 * np.pi * t / 70) + rng.normal(0, 5, t.size)
    x = np.clip(x, 40, 400)

frame, info = build_frame(x)
print(f"Muestras de glucosa: {info['n']} | simbolos por trama: {info['n_symbols']} "
      f"| muestras por trama: {info['frame_len']} | duracion: {info['frame_len'] / FS * 1e3:.3f} ms")
print(f"Paso de cuantizacion: {(info['x_max'] - info['x_min']) / 63:.2f} mg/dL\n")
print(f"{'SNR (dB)':>8} {'EVM %':>7} {'SER':>8} {'BER':>9} {'r':>7} {'NMSE %':>8} {'err max':>8} meta")
for snr in (40, 30, 25, 20, 15):
    dec = decode_frame(canal(frame, snr))
    m = link_metrics(info, dec)
    print(f"{snr:>8} {m['evm_pct']:>7.2f} {m['ser']:>8.4f} {m['ber']:>9.2e} {m['pearson_r']:>7.4f} "
          f"{m['nmse_pct']:>8.4f} {m['max_abs_err']:>8.2f} {'ok' if m['meta_ok'] else 'MAL'}")
print(f"\nCorrelacion solo por cuantizacion (limite sin errores): {m['r_quantization_only']:.4f}")
