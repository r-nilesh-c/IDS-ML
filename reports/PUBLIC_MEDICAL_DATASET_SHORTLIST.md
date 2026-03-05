# Public Medical Dataset Shortlist for Multimodal IDS

## Recommended Primary Source

### 1) BIDMC PPG and Respiration Dataset (PhysioNet, Open Access)
- URL: https://physionet.org/content/bidmc/1.0.0/
- Why it fits: includes numerics such as HR, RR, SpO2 and physiological signals.
- Access: open access, downloadable ZIP available on dataset page.
- Best use in this project: pair with CIC network rows via time/index alignment, then train/evaluate multimodal IDS.

## Strong but Heavy Option

### 2) MIMIC-III Waveform Database (PhysioNet)
- URL: https://physionet.org/content/mimic3wdb/1.0/
- Why it fits: large-scale waveform + numerics (HR, RR, SpO2, blood pressure variants).
- Caution: very large footprint (TB-scale) and operational complexity.
- Best use: advanced follow-up validation, not first pass.

## Optional Auxiliary Source

### 3) MHEALTH Dataset (UCI)
- URL: https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- Why useful: time-series wearable physiological/motion data, easy access.
- Limitation: does not natively provide full bedside vital set (e.g., SpO2/BP), so partial signal coverage.

## Practical Recommendation
- Start with BIDMC + CIC for your main multimodal benchmark.
- Keep MIMIC-III as future extension due to scale.
- Use MHEALTH only as supplementary/ablation source.
