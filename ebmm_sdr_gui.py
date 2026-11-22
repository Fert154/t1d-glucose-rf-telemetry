import sys
import subprocess
import importlib
import site

def _asegurar(modulo, paquete):
    try:
        return importlib.import_module(modulo)
    except ImportError:
        print(f"Instalando {paquete} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", paquete])
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.append(user_site)
        importlib.invalidate_caches()
        return importlib.import_module(modulo)

for _mod, _pkg in [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                   ("PIL", "pillow")]:
    _asegurar(_mod, _pkg)

if sys.platform == "win32":
    import ctypes.util as _cu
    _find_library_original = _cu.find_library

    def _find_library_windows(name):
        found = _find_library_original(name)
        if found is None and name == "c":
            return "msvcrt"
        return found

    _cu.find_library = _find_library_windows
    for _m in [m for m in list(sys.modules) if m == "iio" or m.startswith("adi")]:
        del sys.modules[_m]

try:
    _asegurar("adi", "pyadi-iio")
except (OSError, TypeError) as _e:
    raise SystemExit(
        "\nFalta el controlador libiio de Analog Devices (no se instala con pip).\n"
        "Descarga el instalador de Windows de https://github.com/analogdevicesinc/libiio/releases,\n"
        "instalalo, reinicia Spyder y vuelve a correr este archivo.\n"
        f"Detalle: {_e}")

import tkinter as tk
from PIL import ImageGrab
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import odeint
from scipy.signal import firwin, lfilter
from PIL import ImageGrab, Image
from scipy.signal import correlate, find_peaks
import adi
from scipy.integrate import solve_ivp
from qam64_link import (build_frame, decode_frame, link_metrics,
                      ideal_points, R_DEFAULT, DAC_SCALE)
import gc

t = None
X1 = X2 = X3 = X4 = None
sdr = None
uri_pluto = "ip:analog"

tx_signal = None
tx_info = None
rx_raw = None
rx_result = None
link_result = None
rx_data_corrected = None

def detect_intervals(x, t, threshold=180):
    mask = x > threshold
    d    = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if len(ends) < len(starts):
        ends = np.r_[ends, len(mask)]

    pairs = []
    for s, e in zip(starts, ends):
        t0 = t[s]
        t1 = t[e-1]
        pairs.append((t0, t1))
    return pairs

def take_screenshot(dpi=1200, upsample=True):
    GUI.update()
    x = GUI.winfo_rootx()
    y = GUI.winfo_rooty()
    w = GUI.winfo_width()
    h = GUI.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))

    if upsample:
        scale = dpi / 96
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, resample=Image.LANCZOS)

    filename = f"screenshot_{dpi}dpi.png"
    img.save(
        filename,
        format="PNG",
        dpi=(dpi, dpi),
        optimize=True,
        compress_level=1
    )
    print(f"Guardado como {filename} con {dpi} dpi")

def load_csv():
    global t, X1, X2, X3, X4, hyper_intervals
    fname = filedialog.askopenfilename(
        title="Open CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not fname:
        return
    data = np.loadtxt(fname, delimiter=',', usecols=0)
    t = np.arange(len(data))
    X1 = data.copy()
    X2 = X3 = X4 = np.zeros_like(X1)
    selected_variable.set('x1')
    update_plot()
    max_glucosa_label.config(text=f"Peak Glucose: {X1.max():.1f}")
    messagebox.showinfo("Load CSV", f"{len(data)} Samples loaded from: \n{fname}")

    hyper_intervals = detect_intervals(X1, t, threshold=180)
    intervals_str = [f"{t0:.0f}–{t1:.0f}" for t0, t1 in hyper_intervals]
    if intervals_str:
        texto = "Hyperglycemia:\n" + ", ".join(intervals_str) + " min"
    else:
        texto = "Hyperglycemia: None"
    hyper_time_label.config(text=texto)

    hyper_intervals = detect_intervals(X1, t, threshold=180)
    if hyper_intervals:
        shade_cb.grid(column=6, row=1, pady=(5,0))
    else:
        shade_cb.grid_remove()
    update_plot()
def stop_transmission():
    global sdr
    try:
        if sdr is not None:
            try:
                sdr.tx_cyclic_buffer = False
            except Exception:
                pass

            try:
                sdr.tx(np.zeros(1024, dtype=np.complex64))
            except Exception:
                pass

            try:
                del sdr
                gc.collect()
                sdr = None
                messagebox.showinfo("Transmit", "Transmission stopped and SDR released.")
            except Exception as e:
                messagebox.showerror("SDR Stop Error", f"Error al liberar SDR: {str(e)}")
    except Exception as e:
        messagebox.showerror("SDR Stop Error", str(e))

def on_close():
    stop_transmission()
    GUI.destroy()

def golay_sequences(N: int):
    if N == 1:
        return np.array([1]), np.array([1])
    A, B = golay_sequences(N // 2)
    return np.concatenate([A, B]), np.concatenate([A, -B])

def map_to_64qam_idx(idx):
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    I = levels[(idx // 8) % 8]
    Q = levels[idx % 8]
    return (I + 1j * Q) / np.sqrt(42)

def map_to_64qam(x_norm):
    idx = np.floor(x_norm * 63).astype(int)
    return map_to_64qam_idx(idx)
t = None
X1 = X2 = X3 = X4 = None
sdr = None
uri_pluto = "ip:analog"

tx_signal = None
tx_info = None
rx_raw = None
rx_result = None
link_result = None
rx_data_corrected = None

def run_simulation():
    global t, X1, X2, X3, X4
    try:
        p1, p2, p3 = map(float, (p1_entry.get(), p2_entry.get(), p3_entry.get()))
        p4, p5     = map(float, (p4_entry.get(), p5_entry.get()))
        Gb, Ib     = map(float, (Gb_entry.get(), Ib_entry.get()))
        t_max_val  = float(t_max_entry.get())
        x0         = list(map(float, (x0_1_entry.get(), x0_2_entry.get(),
                                      x0_3_entry.get(), x0_4_entry.get())))

        def model(t, x):
            dx1 = -p1*(x[0] - Gb) - x[0]*x[1] + x[3]
            dx2 = -p2*x[1] + p3*(x[2] - Ib)
            dx3 = -p4*(x[2] - Ib)
            dx4 = -p5*x[3]
            return [dx1, dx2, dx3, dx4]

        t = np.linspace(0, t_max_val, 1014)

        dt = t_max_val/(1500-1)
        sol = solve_ivp(
            fun=model,
            t_span=(0, t_max_val),
            y0=x0,
            method='RK45',
            t_eval=t,
            max_step=dt,
            rtol=1e-6, atol=1e-9
            )

        X1, X2, X3, X4 = sol.y

        update_plot()
        peak = X1.max()
        max_glucosa_label.config(text=f"Peak Glucose: {peak:.1f}")
        intervals = detect_intervals(X1, t, threshold=180)
        intervals_str = [f"{t0:.0f}–{t1:.0f}" for t0, t1 in intervals]
        if intervals_str:
            texto = "Hyperglycemia:\n" + ", ".join(intervals_str) + " min"
        else:
            texto = "Hyperglycemia: None"
        hyper_time_label.config(text=texto)
        idx = np.argmax(X1 > 180)
        hyper = t[idx] if (idx > 0 or X1[0] > 180) else None
        hyper_time_label.config(
            text=f"Hyperglycemia at: {hyper:.1f}" if hyper else "Hyperglycemia at: N/A"
        )
        global hyper_intervals
        hyper_intervals = detect_intervals(X1, t, threshold=180)

        if hyper_intervals:
            shade_cb.grid(column=6, row=1, pady=(5,0))
        else:
            shade_cb.grid_remove()

        update_plot()
    except ValueError:
        messagebox.showerror("Error", "Valores numéricos inválidos.")
def update_plot():
    if t is None: return
    ax.clear()
    var  = selected_variable.get()
    data = {'x1': X1, 'x2': X2, 'x3': X3, 'x4': X4}[var]
    ylab, ttl = {
        'x1':("Glucose (mg/dL)","Glucose simulation"),
        'x2':("Insulin (µU/mL)","Insulin simulation"),
        'x3':("Interstitial Ins (µU/mL)","Interstitial simulation"),
        'x4':("Exogenous Gluc (mg/dL)","Exogenous simulation")
    }[var]
    ax.plot(t, data, label=ylab)
    if var == 'x1':
        ax.axhline(180,c='r',ls='--'); ax.axhline(70,c='b',ls='--')
    ax.set_xlabel("Time (min)"); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.grid(True, ls='--'); ax.legend();
    if shade_var.get() and hyper_intervals:
        for (t0, t1) in hyper_intervals:
            ax.axvspan(t0, t1, color='red', alpha=0.2)
    canvas.draw()

def export_data():
    try:
        fname = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg"),("EPS","*.eps")])
        if not fname: return
        old = fig.get_size_inches()
        fig.set_size_inches(12,6.75,forward=False)
        fig.savefig(fname,dpi=900,bbox_inches='tight')
        fig.set_size_inches(old,forward=False)

        if save_tx_var.get() and tx_signal is not None:
            ftx = filedialog.asksaveasfilename(defaultextension=".csv",
                  filetypes=[("CSV","*.csv")], title="Save TX vector")
            if ftx:
                np.savetxt(ftx, np.c_[tx_signal.real, tx_signal.imag],
                           delimiter=',', fmt="%.6f")
        if save_rx_var.get() and rx_data_corrected is not None:
            frx = filedialog.asksaveasfilename(defaultextension=".csv",
                  filetypes=[("CSV","*.csv")], title="Save RX vector")
            if frx:
                np.savetxt(frx, np.c_[rx_data_corrected.real, rx_data_corrected.imag],
                           delimiter=',', fmt="%.6f")
        messagebox.showinfo("Export","Export completed.")
    except Exception as e:
        messagebox.showerror("Export Error", str(e))

def transmit_data():
    global tx_signal, tx_info, sdr
    try:
        if t is None:
            messagebox.showwarning("Warning", "Run simulation or load a CSV first.")
            return
        data = {'x1': X1, 'x2': X2, 'x3': X3, 'x4': X4}[selected_variable.get()]

        cf = float(center_freq_entry.get()) * 1e9
        sr = float(sample_rate_entry.get()) * 1e6
        bw = float(rf_bw_entry.get()) * 1e6
        gtx = float(tx_gain_entry.get())
        buf = int(buffer_size_entry.get())

        frame, info = build_frame(data, R=R_DEFAULT, scale=DAC_SCALE)

        need = 2 * info["frame_len"] + 64 * R_DEFAULT
        if buf < need:
            buf = int(np.ceil(need / 1024) * 1024)
            buffer_size_entry.delete(0, tk.END)
            buffer_size_entry.insert(0, str(buf))
        elif buf % 1024:
            buf += 1024 - buf % 1024
            buffer_size_entry.delete(0, tk.END)
            buffer_size_entry.insert(0, str(buf))

        tx_signal, tx_info = frame, info

        if sdr is None:
            sdr = adi.ad9361(uri=uri_pluto)
            try:
                sdr._ctx.set_attr("calibrate", "full")
            except Exception:
                pass
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        sdr.sample_rate = int(sr)
        sdr.tx_enabled_channels = [1]
        sdr.rx_enabled_channels = [1]
        sdr.tx_rf_bandwidth = int(bw)
        sdr.tx_lo = int(cf)
        sdr.tx_hardwaregain_chan0 = int(gtx)
        sdr.rx_lo = int(cf)
        sdr.rx_rf_bandwidth = int(bw)
        sdr.rx_buffer_size = buf
        sdr.gain_control_mode_chan0 = 'manual'
        sdr.rx_hardwaregain_chan0 = 0
        sdr.tx_cyclic_buffer = True
        sdr.tx(tx_signal)

        rs = sr / R_DEFAULT
        messagebox.showinfo(
            "Transmit",
            f"Transmitting in loop.\n\n"
            f"Samples of the signal: {info['n']}\n"
            f"Range sent in metadata: {info['x_min']:.4g} to {info['x_max']:.4g}\n"
            f"Symbols per frame: {info['n_symbols']} (32 preamble + 13 metadata + {info['n']} data)\n"
            f"Samples per frame: {info['frame_len']} ({info['frame_len'] / sr * 1e3:.3f} ms)\n"
            f"Symbol rate: {rs / 1e6:.3f} Msym/s  |  Gross bit rate: {6 * rs / 1e6:.2f} Mbit/s\n"
            f"RX buffer: {buf} samples")
    except Exception as e:
        messagebox.showerror("SDR Error", str(e))

def receive_decode():
    global rx_data_corrected, rec_trim, rx_raw, rx_result, link_result
    if sdr is None or tx_info is None:
        messagebox.showwarning("Warning", "Transmit first.")
        return
    try:
        rx_raw = np.asarray(sdr.rx())
        dec = decode_frame(rx_raw, R=R_DEFAULT)
        met = link_metrics(tx_info, dec)
        rx_result, link_result = dec, met
        rx_data_corrected = dec["data_sym"]
        rec_trim = dec["x_hat"]

        plt.figure(figsize=(10, 3))
        plt.plot(dec["corr"], lw=0.8)
        plt.axvline(dec["start"], color="r", lw=1.2, label="Frame start")
        plt.xlabel("Lag [samples]"); plt.ylabel("|Correlation|")
        plt.title("Golay preamble detection"); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.show()

        pts = ideal_points()
        plt.figure(figsize=(6, 6))
        plt.scatter(dec["data_sym"].real, dec["data_sym"].imag, s=6, alpha=0.6, label="Received symbols")
        plt.scatter(pts.real, pts.imag, s=25, marker="+", color="k", label="Ideal 64-QAM")
        plt.axis("equal"); plt.grid(True)
        plt.xlabel("In-phase"); plt.ylabel("Quadrature")
        plt.title(f"Received constellation (EVM = {met['evm_pct']:.2f} %)")
        plt.legend(); plt.tight_layout(); plt.show()

        n = min(len(tx_info["x"]), len(dec["x_hat"]))
        plt.figure(figsize=(10, 4))
        plt.plot(tx_info["x"][:n], lw=1.2, label="TX signal")
        plt.plot(dec["x_hat"][:n], lw=1.2, label="RX signal")
        plt.xlabel("Samples [n]"); plt.ylabel("Value")
        plt.title(f"r = {met['pearson_r']:.4f}  |  NMSE = {met['nmse_pct']:.4f} %  |  SER = {met['ser']:.2e}")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        np.savetxt("rx_resultados.csv",
                   np.c_[np.arange(n), tx_info["x"][:n], tx_info["x_quantized"][:n], dec["x_hat"][:n]],
                   delimiter=",", header="n,tx,tx_quantized,rx", comments="", fmt="%.6g")
        np.savetxt("rx_simbolos.csv", np.c_[dec["data_sym"].real, dec["data_sym"].imag],
                   delimiter=",", header="I,Q", comments="", fmt="%.6f")
        with open("rx_metricas.txt", "w", encoding="utf-8") as f:
            for k, v in met.items():
                f.write(f"{k}: {v}\n")
            f.write(f"cfo_estimated: {dec['cfo_estimated']}\n")
            f.write(f"cfo_hz: {dec['cfo_hz_per_fs'] * float(sample_rate_entry.get()) * 1e6:.2f}\n")
            f.write(f"frame_len: {dec['frame_len']}\n")

        messagebox.showinfo(
            "Received",
            f"Metadata OK: {met['meta_ok']}\n"
            f"Range received: {dec['x_min']:.4g} to {dec['x_max']:.4g}  |  N = {dec['n']}\n\n"
            f"EVM (RMS): {met['evm_pct']:.3f} %\n"
            f"SER: {met['ser']:.2e}   BER: {met['ber']:.2e} ({met['bit_errors']} of {met['n_bits']} bits)\n"
            f"Correlation r: {met['pearson_r']:.4f}  (quantization limit {met['r_quantization_only']:.4f})\n"
            f"NMSE: {met['nmse_pct']:.4f} %\n"
            f"Quantization step: {met['quant_step']:.4g}\n\n"
            f"Saved: rx_resultados.csv, rx_simbolos.csv, rx_metricas.txt")
    except Exception as e:
        messagebox.showerror("RX Error", str(e))

GUI = tk.Tk()
GUI.title("")
GUI.state('zoomed')

GUI.configure(background='#DDDDDD')

style = ttk.Style()
style.theme_use('default')

style.configure('TLabelframe',   background='#DDDDDD', bordercolor='#AAAAAA')
style.configure('TLabelframe.Label', background='#DDDDDD')

style.configure('TFrame',        background='#DDDDDD')

style.configure('TLabel',        background='#DDDDDD')

control = ttk.LabelFrame(GUI, text="System Configuration", padding=(20,10))
control.pack(fill='x', padx=20, pady=10)

param = ttk.LabelFrame(control, text="System Parameters", padding=(10,10))
param.grid(column=0, row=1, padx=10, sticky='nw')

init = ttk.LabelFrame(control, text="Initial Conditions", padding=(10,10))
init.grid(column=1, row=1, padx=10, sticky='nw')

sdrf = ttk.LabelFrame(control, text="SDR Tx Settings", padding=(20,10))
sdrf.grid(column=2, row=1, padx=10, sticky='nw')

def add(lbl, defv, row, frame=param, w=6):
    ttk.Label(frame, text=lbl, font='helvetica 15 bold')\
        .grid(column=0, row=row, sticky='w')
    e = ttk.Entry(frame, font='helvetica 15', width=w)
    e.insert(0, defv)
    e.grid(column=1, row=row, padx=5, pady=2)
    return e

p1_entry = add('Glucose Effectiveness Factor (p1):', 0,    0)
p2_entry = add('Delay in Insulin Action (p2):',         0.015, 1)
p3_entry = add('Insulin Efficacy in Glucose Disposal (p3):', 2e-6,  2)
p4_entry = add('Insulin Degradation rate (p4):',        0.021,  3)
p5_entry = add('Meal Disturbance (p5):',                0.05,   4)
Gb_entry = add('Basal Plasma Glucose (Gb):',            4.5,    5)
Ib_entry = add('Basal Plasma Insulin (Ib):',            4.5,    6)

x0_1_entry = add('Initial Glucose :',       260, 0, init)
x0_2_entry = add('Initial Insulin :',      0.002, 1, init)
x0_3_entry = add('Initial Interstitial Insulin:', 0, 2, init)
x0_4_entry = add('Initial Exogenous Glucose:',   10, 3, init)

def adds(lbl, val, row):
    ttk.Label(sdrf, text=lbl, font='helvetica 15 bold')\
        .grid(column=0, row=row, sticky='w')
    e = ttk.Entry(sdrf, font='helvetica 15', width=10)
    e.insert(0, val)
    e.grid(column=1, row=row, padx=5, pady=2)
    return e

center_freq_entry = adds('Center freq [GHz]:', 2.4, 0)
sample_rate_entry = adds('Sample rate [MSps]:', 61.44, 1)
rf_bw_entry = adds('RF bandwidth [MHz]:', 61.44, 2)
tx_gain_entry      = adds('TX gain [dB]:',       0,        3)
buffer_size_entry  = adds('Buffer size:',      131072,    4)

btn = ttk.Frame(GUI)
btn.pack(pady=10)
hyper_intervals = []

ttk.Label(btn, text="Max sim time:", font='helvetica 15 bold')\
    .grid(column=0, row=0, sticky='e', padx=(10,5))
t_max_entry = ttk.Entry(btn, font='helvetica 15', width=6)
t_max_entry.insert(0, 220)
t_max_entry.grid(column=1, row=0, padx=(0,20))
ttk.Button(btn, text="Screenshot", command=take_screenshot).grid(column=1, row=2, padx=8)
ttk.Button(btn, text="Load CSV",        command=load_csv)     .grid(column=2, row=0, padx=8)
ttk.Button(btn, text="Run Simulation",  command=run_simulation).grid(column=3, row=0, padx=8)
ttk.Button(btn, text="Transmit Data",   command=transmit_data) .grid(column=4, row=0, padx=8)
ttk.Button(btn, text="Receive & Decode",command=receive_decode).grid(column=5, row=0, padx=8)
ttk.Button(btn, text="Export Graph / Data",command=export_data) .grid(column=6, row=0, padx=8)
tk.Button(btn, text="Stop Transmission", command=stop_transmission).grid(column=7, row=0, padx=8)

save_tx_var = tk.BooleanVar()
save_rx_var = tk.BooleanVar()
ttk.Checkbutton(btn, text="Save TX vector", variable=save_tx_var).grid(column=3, row=1)
ttk.Checkbutton(btn, text="Save RX vector", variable=save_rx_var).grid(column=4, row=1)
shade_var = tk.BooleanVar(value=False)

ttk.Checkbutton(
    btn,
    text="Shade Hyperglycemia",
    variable=shade_var,
    command=update_plot
).grid(column=5, row=1, padx=(10,0))
style = ttk.Style()
style.configure('Large.TRadiobutton',
                font=('Helvetica', 16),
                padding=(5, 5))

sel = ttk.Frame(GUI)
sel.pack(anchor='w', padx=60)

selected_variable = tk.StringVar(value='x1')
for txt, val in [("Glucose (x1)", 'x1'), ("Insulin (x2)", 'x2'),
                 ("Interstitial (x3)", 'x3'), ("Exogenous (x4)", 'x4')]:
    ttk.Radiobutton(
        sel,
        text=txt,
        variable=selected_variable,
        value=val,
        command=update_plot,
        style='Large.TRadiobutton'
    ).pack(anchor='w', pady=2)

max_glucosa_label = ttk.Label(GUI, text="Peak Glucose: N/A", font=("Helvetica",18))
max_glucosa_label.pack()
hyper_time_label  = ttk.Label(GUI, text="Hyperglycemia at: N/A", font=("Helvetica",18))
hyper_time_label.pack()

fig, ax = plt.subplots(figsize=(6,3.5))
fig.patch.set_facecolor('#DDDDDD')
ax.set_facecolor('#DDDDDD')
canvas = FigureCanvasTkAgg(fig, master=GUI)
canvas.get_tk_widget().pack(fill='both', expand=True)

GUI.protocol("WM_DELETE_WINDOW", on_close)
GUI.mainloop()
