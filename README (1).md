# Data

The file `combined_data.xlsx` used in the article was built by the authors from the continuous glucose
monitoring (CGM) data described in:

> Fox, I., Ang, L., Jaiswal, M., Pop-Busui, R., & Wiens, J. (2018). Deep multi-output forecasting:
> Learning to accurately predict blood glucose trajectories. *Proceedings of the 24th ACM SIGKDD
> International Conference on Knowledge Discovery & Data Mining*, 1387-1395.
> https://doi.org/10.1145/3219819.3220102

The raw data belong to the original study. `combined_data.xlsx` organizes them into sessions of 1014
consecutive readings per patient, with missing readings interpolated (see below). If the file is not in
this folder, request it from the corresponding author.

## Expected format

The scripts read an Excel file, `combined_data.xlsx`, with a sheet named `Combined Data`:

| Column 1 | Columns 2 ... 1015 |
|---|---|
| Patient identifier | 1014 consecutive CGM readings of one session (mg/dL, every 5 min) |

- One row per sensor session: 408 sessions from 41 patients, with 5 to 13 sessions per patient.
- Values are bounded between 40 and 400 mg/dL by the sensor range.
- Missing readings within each session were filled by linear interpolation (`pandas.DataFrame.interpolate`)
  before the windows were generated.
- Any window that still contains a missing value is discarded by the evaluation script.
- The patient identifier is used to group the cross-validation folds, so all sessions of a patient must
  share the same identifier.
