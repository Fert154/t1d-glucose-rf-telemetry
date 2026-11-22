# T1D glucose prediction and 64-QAM RF telemetry

Code and results for the article:

> F. Torres-Velazco, P. J. Campos, C. Hurtado-Sánchez, E. Inzunza-González, and J. R. Cárdenas-Valdez,
> "An End-to-End Telemedicine System for Predictive Glucose Analytics and RF Telemetry Monitoring
> in Type 1 Diabetes for Internet-Limited Settings," *Sensors* (MDPI), 2026.

| File | Purpose |
|---|---|
| `cross_validation_forecasting.py` | Patient-grouped five-fold cross-validation (Tables 2-5) |
| `fig10_prediction_segments.py` | Figure 10 |
| `fig11_confusion_matrix.py` | Figure 11 |
| `mdpi_figure_style.py` | Common figure style |
| `ebmm_sdr_gui.py` | GUI: EBMM simulation and AD9361 transmission/reception |
| `qam64_link.py` | 64-QAM transmitter and receiver (Algorithms 1-4) |
| `simulated_link_test.py` | Link test over a simulated channel (no hardware) |
| `combined_data.xlsx` | CGM sessions derived from the data of Fox et al. (2018): 408 sessions of 1014 readings from 41 patients |
| `*.csv`, `entorno.txt` | Results reported in the article |

## Usage

```
python -m pip install -r requirements.txt
python cross_validation_forecasting.py
python ebmm_sdr_gui.py
```

The SDR functions also require the Analog Devices libiio driver. 
