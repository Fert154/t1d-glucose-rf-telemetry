# T1D glucose prediction and 64-QAM RF telemetry

Code and results for the article:

> F. Torres-Velazco, P. J. Campos, C. Hurtado-Sánchez, E. Inzunza-González, and J. R. Cárdenas-Valdez,
> "An End-to-End Telemedicine System for Predictive Glucose Analytics and RF Telemetry Monitoring
> in Type 1 Diabetes for Internet-Limited Settings," *Sensors* (MDPI), 2026.

| Script | Purpose |
|---|---|
| `evaluation/cross_validation_forecasting.py` | Patient-grouped five-fold cross-validation (Tables 2-5) |
| `figures/fig10_prediction_segments.py` | Figure 10 |
| `figures/fig11_confusion_matrix.py` | Figure 11 |
| `gui/ebmm_sdr_gui.py` | GUI: EBMM simulation and AD9361 transmission/reception |
| `gui/qam64_link.py` | 64-QAM transmitter and receiver (Algorithms 1-4) |
| `gui/simulated_link_test.py` | Link test over a simulated channel (no hardware) |

`data/combined_data.xlsx` was derived from the CGM data of Fox et al. (2018). `results/` contains
the outputs reported in the article.

## Usage

```
python -m pip install -r requirements.txt
python evaluation/cross_validation_forecasting.py
python gui/ebmm_sdr_gui.py
```

The SDR functions also require the Analog Devices libiio driver. Seed: 42. License: MIT.
