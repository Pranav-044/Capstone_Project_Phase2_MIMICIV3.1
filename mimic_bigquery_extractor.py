"""
mimic_bigquery_extractor.py
============================
Extracts ICU mortality features from MIMIC-IV v3.1 (Google BigQuery)
and saves them as a local CSV ready for Phase 2 training.

WHAT THIS SCRIPT DOES:
  1. Connects to BigQuery using your Google credentials
  2. Queries chartevents (vitals), labevents (labs), admissions (mortality label)
  3. Extracts first-24-hour aggregates per ICU stay
  4. Keeps 30,000 ICU stays (~4,000-7,000 per simulated hospital client)
  5. Saves to mimic_iv_phase2_features.csv

HOW TO RUN:
  python mimic_bigquery_extractor.py --project mimic-analysis-510114

DATASET LOCATION ON BIGQUERY:
  physionet-data.mimiciv_3_1_icu   -> chartevents, icustays, d_items
  physionet-data.mimiciv_3_1_hosp  -> admissions, patients, labevents
"""

import argparse
import os
import pandas as pd
from google.cloud import bigquery

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

ICU_DATASET  = 'physionet-data.mimiciv_3_1_icu'
HOSP_DATASET = 'physionet-data.mimiciv_3_1_hosp'

# How many ICU stays to pull (30k = sweet spot for your 5-client setup)
N_STAYS = 30_000

# First-N-hour window for feature extraction (24h matches MIMIC-IV standard)
OBSERVATION_HOURS = 24

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), 'mimic_iv_phase2_features.csv')


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Get valid ICU stays (min 24h, adults, first ICU stay only)
# ─────────────────────────────────────────────────────────────────────────────

QUERY_STAYS = f"""
WITH stay_info AS (
  SELECT
    i.stay_id,
    i.subject_id,
    i.hadm_id,
    i.intime,
    i.outtime,
    DATETIME_DIFF(i.outtime, i.intime, HOUR) AS los_hours,
    p.anchor_age + (EXTRACT(YEAR FROM i.intime) - p.anchor_year) AS age,
    p.gender,
    a.hospital_expire_flag AS in_hospital_death,
    a.discharge_location,
    ROW_NUMBER() OVER (
      PARTITION BY i.subject_id
      ORDER BY i.intime ASC
    ) AS rn   -- first ICU stay only
  FROM `{ICU_DATASET}.icustays`   i
  JOIN `{HOSP_DATASET}.admissions` a ON i.hadm_id = a.hadm_id
  JOIN `{HOSP_DATASET}.patients`   p ON i.subject_id = p.subject_id
  WHERE
    (p.anchor_age + (EXTRACT(YEAR FROM i.intime) - p.anchor_year)) >= 18  -- adults only
    AND DATETIME_DIFF(i.outtime, i.intime, HOUR) >= {OBSERVATION_HOURS}
)
SELECT *
FROM stay_info
WHERE rn = 1
LIMIT {N_STAYS}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Extract vital signs from chartevents (first 24h)
# ─────────────────────────────────────────────────────────────────────────────

VITAL_ITEMIDS = {
    'heart_rate':         '220045',
    'sbp':                '220179',
    'dbp':                '220180',
    'map':                '220052',
    'resp_rate':          '220210',
    'spo2':               '220277',
    'temperature':        '223761',
    'gcs_total':          '223900',
    'gcs_motor':          '223901',
    'gcs_verbal':         '223900',
    'gcs_eye':            '220739',
    'urine_output_24h':   '226559',
    'weight':             '224639',
    'height':             '226730',
}

VITAL_ITEMID_STR = ', '.join(VITAL_ITEMIDS.values())

QUERY_VITALS_FAST = f"""
WITH valid_stays AS (
  SELECT i.stay_id, i.intime
  FROM `{ICU_DATASET}.icustays` i
  JOIN `{HOSP_DATASET}.admissions` a ON i.hadm_id = a.hadm_id
  JOIN `{HOSP_DATASET}.patients`   p ON i.subject_id = p.subject_id
  WHERE (p.anchor_age + (EXTRACT(YEAR FROM i.intime) - p.anchor_year)) >= 18
    AND DATETIME_DIFF(i.outtime, i.intime, HOUR) >= {OBSERVATION_HOURS}
  QUALIFY ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime ASC) = 1
  LIMIT {N_STAYS}
)
SELECT
  c.stay_id,
  c.itemid,
  AVG(c.valuenum)    AS value_mean,
  MIN(c.valuenum)    AS value_min,
  MAX(c.valuenum)    AS value_max,
  STDDEV(c.valuenum) AS value_std
FROM `{ICU_DATASET}.chartevents` c
JOIN valid_stays vs ON c.stay_id = vs.stay_id
WHERE
  c.itemid IN ({VITAL_ITEMID_STR})
  AND c.valuenum IS NOT NULL
  AND c.valuenum > 0
  AND DATETIME_DIFF(c.charttime, vs.intime, HOUR) BETWEEN 0 AND {OBSERVATION_HOURS}
GROUP BY c.stay_id, c.itemid
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: Extract lab values from labevents (first 24h)
# ─────────────────────────────────────────────────────────────────────────────

LAB_ITEMIDS = {
    'bun':          '51006',
    'creatinine':   '50912',
    'glucose':      '50931',
    'sodium':       '50983',
    'potassium':    '50971',
    'hemoglobin':   '51222',
    'wbc':          '51301',
    'platelets':    '51265',
    'bicarbonate':  '50882',
    'lactate':      '50813',
    'bilirubin':    '50885',
    'albumin':      '50862',
    'troponin':     '51003',
    'ph':           '50820',
    'pco2':         '50818',
    'po2':          '50821',
}

LAB_ITEMID_STR = ', '.join(LAB_ITEMIDS.values())

QUERY_LABS_FAST = f"""
WITH valid_stays AS (
  SELECT i.stay_id, i.hadm_id, i.intime
  FROM `{ICU_DATASET}.icustays` i
  JOIN `{HOSP_DATASET}.admissions` a ON i.hadm_id = a.hadm_id
  JOIN `{HOSP_DATASET}.patients`   p ON i.subject_id = p.subject_id
  WHERE (p.anchor_age + (EXTRACT(YEAR FROM i.intime) - p.anchor_year)) >= 18
    AND DATETIME_DIFF(i.outtime, i.intime, HOUR) >= {OBSERVATION_HOURS}
  QUALIFY ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime ASC) = 1
  LIMIT {N_STAYS}
)
SELECT
  vs.stay_id,
  l.itemid,
  AVG(l.valuenum)    AS value_mean,
  MIN(l.valuenum)    AS value_min,
  MAX(l.valuenum)    AS value_max,
  STDDEV(l.valuenum) AS value_std
FROM `{HOSP_DATASET}.labevents` l
JOIN valid_stays vs ON l.hadm_id = vs.hadm_id
WHERE
  l.itemid IN ({LAB_ITEMID_STR})
  AND l.valuenum IS NOT NULL
  AND l.valuenum > 0
  AND DATETIME_DIFF(l.charttime, vs.intime, HOUR) BETWEEN 0 AND {OBSERVATION_HOURS}
GROUP BY vs.stay_id, l.itemid
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Processing: Pivot and merge into one row per ICU stay
# ─────────────────────────────────────────────────────────────────────────────

VITAL_ID_TO_NAME = {v: k for k, v in VITAL_ITEMIDS.items()}
LAB_ID_TO_NAME   = {v: k for k, v in LAB_ITEMIDS.items()}


def pivot_features(df: pd.DataFrame,
                   itemid_to_name: dict,
                   suffix: str = '') -> pd.DataFrame:
    """Pivot long-format itemid rows into one wide row per stay_id."""
    df = df.copy()
    df['feature_name'] = df['itemid'].astype(str).map(itemid_to_name)
    df = df.dropna(subset=['feature_name'])

    pivoted = df.pivot_table(
        index='stay_id',
        columns='feature_name',
        values=['value_mean', 'value_min', 'value_max'],
        aggfunc='first',
    )
    pivoted.columns = [f"{val}_{col}{suffix}"
                       for val, col in pivoted.columns]
    return pivoted.reset_index()


def build_feature_matrix(stays_df, vitals_df, labs_df) -> pd.DataFrame:
    """Merge stays + vitals + labs into one row per stay."""
    stays_df = stays_df[['stay_id', 'subject_id', 'age', 'gender',
                          'los_hours', 'in_hospital_death']]

    # Pivot vitals and labs
    vitals_wide = pivot_features(vitals_df, VITAL_ID_TO_NAME, suffix='_vital')
    labs_wide   = pivot_features(labs_df,   LAB_ID_TO_NAME,   suffix='_lab')

    merged = stays_df\
        .merge(vitals_wide, on='stay_id', how='left')\
        .merge(labs_wide,   on='stay_id', how='left')

    # Encode gender
    merged['gender_male'] = (merged['gender'] == 'M').astype(int)
    merged = merged.drop(columns=['gender'], errors='ignore')

    # Drop rows with >60% missing values (unreliable stays)
    thresh = int(0.40 * len(merged.columns))
    merged = merged.dropna(thresh=thresh)

    # Fill remaining NaN with column median (standard clinical imputation)
    num_cols = merged.select_dtypes(include='number').columns
    merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())

    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  Main Extraction Function
# ─────────────────────────────────────────────────────────────────────────────

def extract(project_id: str):
    print(f"\n  Connecting to BigQuery project: {project_id}")
    client = bigquery.Client(project=project_id)

    print(f"\n[1/3] Fetching {N_STAYS} valid ICU stays ...")
    stays_df = client.query(QUERY_STAYS).to_dataframe()
    print(f"  Got {len(stays_df)} stays | "
          f"mortality rate: {stays_df['in_hospital_death'].mean():.3f}")

    print(f"\n[2/3] Fetching vital signs (first {OBSERVATION_HOURS}h) ...")
    print("  (This query scans chartevents - may take 1-2 minutes)")
    vitals_df = client.query(QUERY_VITALS_FAST).to_dataframe()
    vitals_df['itemid'] = vitals_df['itemid'].astype(str)
    print(f"  Got {len(vitals_df)} vital-sign rows")

    print(f"\n[3/3] Fetching lab values (first {OBSERVATION_HOURS}h) ...")
    labs_df = client.query(QUERY_LABS_FAST).to_dataframe()
    labs_df['itemid'] = labs_df['itemid'].astype(str)
    print(f"  Got {len(labs_df)} lab-value rows")

    print("\n  Building feature matrix ...")
    feature_df = build_feature_matrix(stays_df, vitals_df, labs_df)

    # Show what we have
    print(f"\n  Final shape: {feature_df.shape}")
    print(f"  Rows (patients):  {len(feature_df)}")
    print(f"  Columns (features+label): {len(feature_df.columns)}")
    print(f"  Mortality rate: {feature_df['in_hospital_death'].mean():.3f}")
    print(f"  Missing values: {feature_df.isnull().sum().sum()} (after imputation)")

    # Save
    feature_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  OK Saved to: {OUTPUT_FILE}")
    print(f"    File size: ~{os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")

    return feature_df


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract MIMIC-IV v3.1 ICU features from Google BigQuery')
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Your Google Cloud Project ID (e.g. mimic-analysis-510114)')
    parser.add_argument(
        '--n_stays',
        type=int,
        default=30_000,
        help='Number of ICU stays to extract (default: 30000)')
    parser.add_argument(
        '--hours',
        type=int,
        default=24,
        help='Observation window in hours (default: 24)')
    args = parser.parse_args()

    # Override globals if CLI args provided
    N_STAYS = args.n_stays
    OBSERVATION_HOURS = args.hours

    print("=" * 60)
    print("  MIMIC-IV v3.1  BigQuery Feature Extractor")
    print("=" * 60)
    print(f"  ICU stays to extract:  {N_STAYS:,}")
    print(f"  Observation window:    first {OBSERVATION_HOURS}h")
    print(f"  Output file:           {OUTPUT_FILE}")

    df = extract(project_id=args.project)

    print("\nDone. You can now run:")
    print("  python main_phase2.py --seed 42 --use_smote --rounds 20 --verbose")
