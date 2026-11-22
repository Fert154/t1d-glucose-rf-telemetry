# End-to-End Telemedicine System for Predictive Glucose Analytics and RF Telemetry in Type 1 Diabetes

Code and results accompanying the article:

> F. Torres-Velazco, P. J. Campos, C. Hurtado-Sánchez, E. Inzunza-González, and J. R. Cárdenas-Valdez,
> "An End-to-End Telemedicine System for Predictive Glucose Analytics and RF Telemetry Monitoring
> in Type 1 Diabetes for Internet-Limited Settings," *Sensors* (MDPI), 2026 (under review).

The repository contains:

1. **Glucose forecasting and glycemic-event classification**: random forest and XGBoost regressors
   evaluated with five-fold cross-validation grouped by patient, at horizons of 15, 30, 45, and 60 min.
2. **64-QAM RF link**: transmitter and receiver for the AD9361 (FMCOMMS2 + ZedBoard) through `pyadi-iio`,
   plus a channel simulation that runs without hardware.
3. **Graphical interface**: Extended Bergman Minimal Model (EBMM) simulation, CSV loading, transmission, and
   reception in a single Python GUI.


## Repository structure

```
evaluation/
    cross_validation_forecasting.py  Patient-grouped cross-validation (Tables 2-5)
figures/
    mdpi_figure_style.py             Common figure style
    fig10_prediction_segments.py     Figure 10: real vs. predicted glucose (24 h segments)
    fig11_confusion_matrix.py        Figure 11: confusion matrix at 45 min
gui/
    ebmm_sdr_gui.py                  GUI: EBMM simulation + SDR transmission/reception
    qam64_link.py                    64-QAM frame construction and receiver (Algorithms 1-4)
    simulated_link_test.py           Link test over a simulated channel (no hardware needed)
data/
    README.md                        Data source and file format
results/
    *.csv, entorno.txt               Results reported in the article
```

## Installation

Python 3.14 was used. Install the dependencies with

```
python -m pip install -r requirements.txt
```

`ebmm_sdr_gui.py` also installs any missing package automatically on first run. The SDR functions additionally
require the **libiio** driver from Analog Devices
(<https://github.com/analogdevicesinc/libiio/releases>); the EBMM simulation and the simulated link work
without it.

## Reproducing the results

### 1. Forecasting and classification (Tables 2-5)

Place the data file in `data/combined_data.xlsx` )
```
cd evaluation
python cross_validation_forecasting.py
```

This runs persistence, random forest, and XGBoost (default and tuned by nested cross-validation) with
seed 42 and writes all tables to `cv_results/`. The run takes several hours on a
12-thread desktop CPU because of the nested hyperparameter search. Options:

| Option | Meaning |
|---|---|
| `--file`, `--sheet` | Data file and sheet (default `../data/combined_data.xlsx`, `Combined Data`) |
| `--solo MODEL ...` | Models to run (`RF-delta`, `RF-delta-tuned`, `XGB-delta`, `XGB-delta-tuned`) |
| `--n-rf`, `--n-xgb` | Random-search candidates (default 12 and 20) |
| `--submuestra` | The search uses one of every N windows (default 4) |
| `--arboles-busqueda` | Random forest trees during the search (default 60) |

Main settings:
- Input: windows of 12 consecutive readings (1 h), stride 1 sample.
- Features: 12 readings, 11 first differences, and the mean, standard deviation, and slope (26 in total).
- Target: the change G(t+H) − G(t).
- Classes: hypoglycemia below τ_H, hyperglycemia above 180 mg/dL. τ_H is calibrated on 25 % of the
  training patients (grid 70-120 mg/dL).
- Every fitted step (hyperparameters and τ_H) uses only the training patients of each fold.

The output files are the ones in `results/`:

| File | Content |
|---|---|
| `tabla_A_regresion.csv` | RMSE, MAE, MARD (mean ± SD over folds) |
| `tabla_B_clases.csv` | Per-class precision, recall, F1 |
| `tabla_C_hipoglucemia.csv` | Hypoglycemia sensitivity, specificity, precision, F1, AUC-PR |
| `pruebas_pareadas.csv` | Wilcoxon (per-patient MAE) and McNemar (per-window class) tests |
| `hiperparametros_elegidos.csv` | Configuration selected in each fold and horizon |
| `pliegues_tamanos.csv` | Patients and windows per fold |
| `tiempos.csv` | Training time and single-window latency |
| `entorno.txt` | Software versions, hardware, and settings of the run |

### 2. Figures 10 and 11

```
cd figures
python fig10_prediction_segments.py   # candidate 24 h segments; the article uses patient 17
python fig11_confusion_matrix.py      # confusion matrix from the counts in results/matrices_confusion.csv
```

### 3. 64-QAM link without hardware

```
cd gui
python simulated_link_test.py [glucose.csv]
```

Transmits a glucose series through a simulated channel. The channel includes random delay, gain and
phase, carrier frequency offset, IQ imbalance, and AWGN. The script then reports EVM, SER, BER, Pearson
correlation, and reconstruction error for several SNR values.

Frame parameters:
- Frame: [32 Golay preamble symbols | 13 metadata symbols | N data symbols].
- Each glucose sample is one 64-QAM symbol (6 bits), and each symbol is repeated R = 32 times.
- At 61.44 MS/s this gives 1.92 Msymbol/s, and one frame is (45 + N)·32 samples.

### 4. GUI with the AD9361

```
cd gui
python ebmm_sdr_gui.py
```

Set the board address in `uri_pluto` at the top of the file (default `ip:analog`). Then:

1. **Run Simulation** or **Load CSV**.
2. **Transmit Data**: cyclic TX.
3. **Receive & Decode**: saves `rx_resultados.csv`, `rx_simbolos.csv`, and `rx_metricas.txt`.



