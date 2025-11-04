import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import os
import json
from sklearn.preprocessing import LabelEncoder
from config import CONFIG
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


import warnings

# Global suppression
warnings.filterwarnings("ignore", message="Parsing dates in .* when dayfirst=True was specified")
warnings.filterwarnings("ignore", category=FutureWarning)


def get_dfs():
    dfs = {}

    # Static data for recepient
    dfs['demographics'] = pd.read_excel('../NephroCAGE/NephroCAGE/1_demographics.xlsx', dtype=str)

    # Static data for donor
    dfs['donorparameter'] = pd.read_excel('../NephroCAGE/NephroCAGE/2_donoparameter_final.xlsx', dtype=str)
    dfs['hospitalization'] = pd.read_excel('../NephroCAGE/NephroCAGE/9_Hospitalization.xlsx', dtype=str)

    dfs['vital_parameters'] = pd.read_csv('../NephroCAGE/NephroCAGE/7_clinical_assessment.csv', delimiter=";", dtype=str, encoding='latin1')

    dfs['lab_values'] = pd.read_csv('../NephroCAGE/NephroCAGE/6_Lab_cohort.csv', delimiter=";", dtype=str, encoding='latin1')
    dfs['medication'] = pd.read_csv('../NephroCAGE/NephroCAGE/8_Medikation.csv', delimiter=";", dtype=str, encoding='latin1')
    dfs['biopsy'] = pd.read_excel('../NephroCAGE/NephroCAGE/5_Biopsy_patho_kreuz.xlsx', dtype=str)

    return dfs

# function to calculate risk of graft failure
# Weighted Risk Score=(1.06 ^ (mma_broad + mmb_broad))*(1.12 ^ mmdr_broad)
def calculate_risk(row):
    return (1.06 ** row['mma_broad']) * (1.01 ** row['mmb_broad']) * (1.12 ** row['mmdr_broad'])

# Function to calculate days in hospital
def calculate_days_in_hospital(dfs):
    # For each patient_id and transplant_id, calculate the days in hospital
    hospitalization = dfs['hospitalization'][['PatientID', 'TransplantationID', 'KHStarttime', 'KHEndtime']].rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'KHStarttime': 'KHStarttime',
            'KHEndtime': 'KHEndtime'
        }
    )
    hospitalization[['KHStarttime', 'KHEndtime']] = hospitalization[['KHStarttime', 'KHEndtime']].apply(pd.to_numeric, errors='coerce')

    # Calculate days in hospital for each row
    hospitalization['days_in_hospital'] = hospitalization['KHEndtime'] - hospitalization['KHStarttime']

    # Sum days per patient and transplant
    hospitalization_summary = hospitalization.groupby(['patient_id', 'transplant_id'])['days_in_hospital'].sum().reset_index()

    return hospitalization_summary

def get_dict(df, column_name):
    return {value: idx for idx, value in enumerate(df[column_name].unique(), start=1)}


def load_json(load_path):
    """
    Loads the value dictionary from a JSON file.
    """
    with open(load_path, "r") as f:
        value_dict = json.load(f)
    return value_dict



def create_static_df(dfs):

    # -----------------------Create static dataframe-------------------------
    static_df = dfs['demographics'][['PatientID','TransplantationID', 'SpenderID', 'Todesdatum', 'Datum', 'Geschlecht', 'Grunderkrankung', 
                          'Datum_erste_Dialyse', 'Dialyse_Anzahl', 'Alter', 'Blutgruppe', 'EBV_IgG','delayed graft function_inverse', 
                          'Koerpergroesse','MMA_broad', 'MMB_broad',
                          'MMDR_broad','MM_broad', 'Date of graft loss', 'Loss cause', 'cold ischemia time', 'PIRCHE_Score']].rename(
        columns={
            'PatientID': 'patient_id',
            'SpenderID': 'donor_id',
            'TransplantationID': 'transplant_id',
            'Todesdatum': 'death_date',
            'Datum': 'transplant_date',
            'Geschlecht': 'gender',
            'Grunderkrankung': 'underlying_disease',
            'Datum_erste_Dialyse': 'first_dialysis_date',
            'Dialyse_Anzahl': 'number_dialyses',
            'Alter': 'age',
            'Blutgruppe': 'blood_group',
            'EBV_IgG': 'ebv_status',
            'delayed graft function_inverse': 'delayed_graft_function',
            'Koerpergroesse': 'height',
            'MMA_broad': 'mma_broad', 
            'MMB_broad': 'mmb_broad',
            'MMDR_broad': 'mmdr_broad',
            'MM_broad': 'mm_broad',
            'Date of graft loss': 'loss_date',
            'Loss cause': 'loss_cause',
            'cold ischemia time': 'cold_ischemia_time',
            'PIRCHE_Score': 'pirche_score'
        }
    )

    #-------- drop nan for patiend id, donor id, transplant id ----------
    # Remove patients without donor_id and transplant_id
    static_df = static_df.dropna(subset=['patient_id', 'donor_id', 'transplant_id'])

    #-------adding number of transplantation-------

    # Calculate number of Transplantations for each patient
    transplantations = static_df.groupby('patient_id')['transplant_id'].count()

    # Add transplantation count to demographics
    static_df['number_transplantation'] = static_df['patient_id'].map(transplantations)

    #--------------------- Map donor to patients -------------------
    static_df = static_df.merge(
        dfs['donorparameter'][['PatientID', 'TransplantationID', 'SpenderID', 'gender_donor', 'age_donor', 'donor_bloodgroup',
                                'type_of_donation']],
        how='left',  # Left join to keep all records in static_df
        left_on=['patient_id', 'transplant_id', 'donor_id'],
        right_on=['PatientID', 'TransplantationID', 'SpenderID']
    )

    static_df = static_df.drop(columns=['PatientID', 'TransplantationID', 'SpenderID'])
    
    # Drop duplicate [patient_ids]
    static_df = static_df.drop_duplicates(subset=['patient_id'])

    # Replace 'u' with NaN in gender_donor column
    static_df['gender_donor'] = static_df['gender_donor'].replace('u', np.nan)

    #--------------- map days_in hospital to static_df ----------------
    static_df = static_df.merge(
        calculate_days_in_hospital(dfs),
        how='left',
        on=['patient_id', 'transplant_id']
    )

    # ---------------- calculate loss rel days and death rel days --------------

    # convert to date time
    static_df['transplant_date'] = pd.to_datetime(static_df['transplant_date'], dayfirst=True, errors='coerce')
    static_df['loss_date'] = pd.to_datetime(static_df['loss_date'], dayfirst=True, errors='coerce')
    static_df['death_date'] = pd.to_datetime(static_df['death_date'], dayfirst=True, errors='coerce')
 
    # if loss date is given calculate the relative days from transplantation date
    static_df['loss_rel_days'] = (static_df['loss_date'] - static_df['transplant_date']).dt.days
    static_df['death_rel_days'] = (static_df['death_date'] - static_df['transplant_date']).dt.days

    # if loss rel_days more than 1825 days , drop this patient
    # static_df = static_df[(static_df['loss_rel_days'] <= 1825) | (static_df['loss_rel_days'].isna())]                                                               

    # ----------- numerical features ---------------------
    
    #static_df['patient_id'] = pd.to_numeric(static_df['patient_id'], errors='coerce')
    #static_df['transplant_id'] = pd.to_numeric(static_df['transplant_id'], errors='coerce')

    
    # make pirche score ready for numerical transformation
    static_df['pirche_score'] = static_df['pirche_score'].str.replace(r'[^0-9.]', '', regex=True)
    static_df['pirche_score'] = pd.to_numeric(static_df['pirche_score'], errors='coerce').round(4)

    numerical_features = ['mma_broad', 'mmb_broad', 'mmdr_broad', 'mm_broad']
    for col in numerical_features:
        static_df[col] = pd.to_numeric(static_df[col], errors='coerce').astype('Int64')   # safely convert


    #------------------ categorical features ---------------------

    # lower case values in delayed graft function and gender donor
    static_df['delayed_graft_function'] = static_df['delayed_graft_function'].str.lower()
    static_df['gender_donor'] = static_df['gender_donor'].str.lower()
    static_df['type_of_donation'] = static_df['type_of_donation'].str.lower()
    static_df['underlying_disease'] = static_df['underlying_disease'].str.lower()
    static_df['loss_cause'] = static_df['loss_cause'].str.lower()

    # Convert standalone blood groups to include +
    static_df.loc[static_df['blood_group'].isin(['A', 'B', 'AB', '0']), 'blood_group'] += '+'
    static_df.loc[static_df['donor_bloodgroup'].isin(['A', 'B', 'AB', '0']), 'donor_bloodgroup'] += '+'
    
    # Set invalid blood groups to NaN
    valid_blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', '0+', '0-']
    static_df.loc[~static_df['blood_group'].isin(valid_blood_groups), 'blood_group'] = np.nan
    static_df.loc[~static_df['donor_bloodgroup'].isin(valid_blood_groups), 'donor_bloodgroup'] = np.nan

    static_df['blood_group'] = static_df['blood_group'].str.lower()
    static_df['donor_bloodgroup'] = static_df['donor_bloodgroup'].str.lower()

    #death and loss labels
    static_df['death_label'] = static_df['death_rel_days'].notna().astype(int)
    static_df['loss_label'] = static_df['loss_rel_days'].notna().astype(int)

    # ----------- text features -------------------

    # fill nan for categorical features with unknown
    for col in CONFIG['static_categorical_feat']:
        static_df[col] = static_df[col].fillna('unknown')

    #print("categorical cardinalities", categorical_cardinalities)

    # -------------- calculate risk --------------------
    # Calculate risk of graft failure for patients
    static_df['risk_of_graft_failure'] = static_df.apply(calculate_risk, axis=1)
    # use geometric mean of pirche_score and risk_of_graft_failure and save as risk_score  
    static_df['risk_score'] = (static_df['pirche_score'] * static_df['risk_of_graft_failure']) ** (1/2)

    # drop Nan for risk_score
    static_df = static_df.dropna(subset=['risk_score'])

    # Calculate mean risk score
    mean_risk = static_df['risk_score'].mean()
    std_risk = static_df['risk_score'].std()  # Standard deviation as threshold

    # Assign risk labels based on mean
    static_df['risk_label'] = static_df['risk_score'].apply(lambda x: 0 if x < mean_risk - std_risk else 
                                                    (1 if x <= mean_risk + std_risk else 2))
    #static_df['risk_label'] = pd.qcut(static_df['risk_score'], q=3, labels=[0, 1, 2])

    
    #print("min max scaler for static features")
    #print('min pirche score', static_df['pirche_score'].min())
    #print('max pirche score', static_df['pirche_score'].max())

    # -------------- drop unuseful columns ---------------------
    # drop 'SOURCE', 'donor_COD', 'pirche_score', 'mma_broad', 'mmb_broad', 'mmdr_broad', 'mm_broad', 'loss_date', 'death_date', 'first_dialysis_date'
    static_df = static_df.drop(columns=['mma_broad', 'mmb_broad', 'mmdr_broad', 'mm_broad',  
                                        'first_dialysis_date', 'loss_cause', 'risk_score', 'loss_date', 'death_date', 'donor_id'])
    return static_df

def create_vitals_df(dfs, static_df=None):

    if static_df is None:
        static_df, _ = create_static_df(dfs)

    # create vitals df
    vital_parameters = dfs['vital_parameters'][['PatientID', 'TransplantationID', 'OPDtime', 'Blutdruck_systolisch', 'Blutdruck_diastolisch', 'Gewicht', 'Urinvolumen', 'Herzfrequenz', 'Temperatur', 'Diuresezeit']].rename(
            columns={
                'PatientID': 'patient_id',
                'TransplantationID': 'transplant_id',
                'OPDtime': 'rel_days', # date of assessment in days after transplantation
                'Blutdruck_systolisch': 'bp_sys',
                'Blutdruck_diastolisch': 'bp_dia',
                'Gewicht': 'weight',
                'Urinvolumen': 'urine_volume',  # lower urine volume indicates issues with kidney function
                'Herzfrequenz': 'heart_rate',
                'Temperatur': 'temperature',
                'Diuresezeit': 'diuresis_time',  # duration over which urine output is measured
            }
    )

    # ------------ change type of patient id and transplant id ------------------
    #vital_parameters['patient_id'] = pd.to_numeric(vital_parameters['patient_id'], errors='coerce')
    #vital_parameters['transplant_id'] = pd.to_numeric(vital_parameters['transplant_id'], errors='coerce')

    # --------------- data normalization ---------------------------
    features_to_clean = ['bp_sys', 'bp_dia', 'weight', 'urine_volume', 'heart_rate', 'temperature', 'diuresis_time']

    # Replace commas with dots and convert to numeric for the specified features
    for feature in features_to_clean:
        vital_parameters[feature] = vital_parameters[feature].astype(str).str.replace(',', '.', regex=False)
        vital_parameters[feature] = pd.to_numeric(vital_parameters[feature], errors='coerce')

    vital_parameters = vital_parameters.merge(
        static_df[['patient_id', 'transplant_id']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    vital_parameters = vital_parameters.dropna(subset=['rel_days'])
    vital_parameters['rel_days'] = vital_parameters['rel_days'].astype(str).str.replace(',', '.')
    vital_parameters['rel_days'] = pd.to_numeric(vital_parameters['rel_days'], errors='coerce').astype(int)
    # Drop rows with NaN in 'rel_days'
    vital_parameters = vital_parameters[(vital_parameters['rel_days'] >= 0)]
    #print("vital data for patient 2327", vital_parameters[vital_parameters['patient_id'] == '2327'])

    #print("min value for vital parameters")
    #print(vital_parameters[features_to_clean].min())
    #print("max value for vital parameters")
    #print(vital_parameters[features_to_clean].max())


    '''
    vital_features_to_scale =  ['bp_sys', 'bp_dia', 'heart_rate', 'temperature', 'urine_volume', 'diuresis_time', 'weight'] 
    scalers = {}
    # Apply scaler
    for col in vital_features_to_scale:# Replace NaNs with 0, or use another method
        scaler = MinMaxScaler()
        vital_parameters[col] = scaler.fit_transform(vital_parameters[[col]])
        scalers[col] = scaler'''
    
    return vital_parameters

def create_lab_values_df(dfs, static_df=None):
    if static_df is None:
        static_df, _ = create_static_df(dfs)

    lab_values = dfs['lab_values'][['PatientID', 'TransplantationID', 'Labtime', 'Bezeichnung', 'Wert', 'Einheit']].rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'Labtime': 'rel_days', # date of assessment in days after transplantation
            'Bezeichnung': 'description',
            'Wert' : 'value',
            'Einheit': 'unit',
        }
    )

    lab_values = lab_values.merge(
        static_df[['patient_id', 'transplant_id']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    # process rel_days
    lab_values['rel_days'] = lab_values['rel_days'].astype(str).str.replace(',', '.').astype(float).astype(int) # change to int
    lab_values = lab_values.dropna(subset=['rel_days'])  # Drop rows with NaN in 'rel_days'
    lab_values = lab_values[(lab_values['rel_days'] >= 0)] # keep  only from day 0 till day 1825(5years)

    lab_filtered = lab_values[lab_values['description'].isin(['KreatininHP', 'LeukoEB', 'CRPHP', 'ProteinCSU', 'AlbuminKSU'])].copy()
    lab_filtered['value'] = pd.to_numeric(lab_filtered['value'].astype(str).str.replace(',', '.'), errors='coerce')
    lab_filtered = lab_filtered.dropna(subset=['value'])
    lab_filtered['unit'] = lab_filtered['unit'].astype(str).str.lower()

    # Convert CRPHP mg/l to mg/dl
    crphp_mask = lab_filtered['description'] == 'CRPHP'
    mg_l_mask = crphp_mask & (lab_filtered['unit'] == 'mg/l')
    lab_filtered.loc[mg_l_mask, 'value'] = lab_filtered.loc[mg_l_mask, 'value'] / 10
    lab_filtered.loc[crphp_mask, 'unit'] = 'mg/dl'

    # Apply filters
    k_mask = (lab_filtered['description'] == 'KreatininHP') & (lab_filtered['unit'].isin(['mg/dl', 'mg/dl'])) & (lab_filtered['value'] <= 50)
    leu_mask = (lab_filtered['description'] == 'LeukoEB') & (lab_filtered['unit'] == '/nl') & (lab_filtered['value'] <= 5000)
    crphp_mask = (lab_filtered['description'] == 'CRPHP') & (lab_filtered['unit'] == 'mg/dl') & (lab_filtered['value'] <= 200)
    protein_mask = (lab_filtered['description'] == 'ProteinCSU') & (lab_filtered['unit'] == 'mg/l') & (lab_filtered['value'] <= 2000)
    alb_mask = (lab_filtered['description'] == 'AlbuminKSU') & (lab_filtered['value'] <= 5000)
    lab_filtered = lab_filtered[k_mask | leu_mask | crphp_mask | protein_mask | alb_mask]


    lab_pivot = lab_filtered.pivot_table(
                index=['patient_id', 'transplant_id', 'rel_days'],
                columns='description',
                values='value',
                aggfunc='first'
    ).reset_index()

    lab_pivot = lab_pivot.rename(columns={'KreatininHP': 'creatinine', 'LeukoEB': 'leukocyte', 'CRPHP': 'crphp', 'ProteinCSU': 'proteinuria', 'AlbuminKSU': 'albumin'})
    #print("lab values for patient 2327", lab_pivot[lab_pivot['patient_id'] == '2327'])
    '''
    lab_features = ['albumin', 'proteinuria', 'leukocyte', 'creatinine', 'crphp']
    
    scalers = {}
    for col in lab_features:
        scaler = MinMaxScaler()
        lab_pivot[col] = scaler.fit_transform(lab_pivot[[col]])
        scalers[col] = scaler'''

    return lab_pivot

def create_medication_df(dfs, static_df=None):
    if static_df is None:
        static_df, _= create_static_df(dfs)

    medication = dfs['medication'][['PatientID', 'TransplantationID', 'prescription start', 'prescription end', 'Bezeichnung', 'DDD', 'unit', 'ATC']].rename(
        columns={
                'PatientID': 'patient_id',
                'TransplantationID': 'transplant_id',
                'prescription start': 'p_start',
                'prescription end': 'p_end',
                'Bezeichnung': 'description',
                'DDD': 'ddd', # defined daily dose
                'unit': 'unit',
                'ATC': 'atc' # the ATC column identifies the medication based on its pharmacological classification.
            }
    )
    
    #medication['patient_id'] = pd.to_numeric(medication['patient_id'], errors='coerce')
    #medication['transplant_id'] = pd.to_numeric(medication['transplant_id'], errors='coerce')
    
    medication = medication.merge(
        static_df[['patient_id', 'transplant_id', 'transplant_date']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    # Calculate the relative days from transplantation date to prescription start and end
    medication['start'] = (pd.to_datetime(medication['p_start'], dayfirst=True)-
                                                pd.to_datetime(medication['transplant_date'], dayfirst=True)
                                                ).dt.days
    medication['end'] = (pd.to_datetime(medication['p_end'], dayfirst=True)-
                                                pd.to_datetime(medication['transplant_date'], dayfirst=True)
                                                ).dt.days
    medication = medication.drop(columns=['p_start', 'p_end', 'transplant_date'])
    medication = medication.dropna(subset=['start', 'end'])

    medication['start'] = medication['start'].astype(str).str.replace(',', '.').astype(float).astype(int)
    medication['end'] = medication['end'].astype(str).str.replace(',', '.').astype(float).astype(int)

    # filter medication
    filtered_atcs = ['L04AD02', 'L04AA06', 'L04AD01']

    medication = medication[medication['atc'].isin(filtered_atcs)]
    # Keep only the rows where the unit is 'mg'
    medication = medication[medication['unit'] == 'mg']

    medication['ddd'] = medication['ddd'].astype(str).str.replace(',', '.')
    medication['ddd'] = pd.to_numeric(medication['ddd'], errors='coerce')
    medication = medication.dropna(subset=['ddd'])
    
    # Remove entries where ddd > 2000, these are erroneous rows
    medication = medication[medication['ddd'] <= 2000]

    medication['rel_days'] = medication.apply(
            lambda r: range(r['start'], r['end'] + 1), axis=1
        )
    medication = medication.explode('rel_days').drop(columns=['start', 'end'])
    
    medication = medication.dropna(subset=['rel_days'])

    medication['rel_days'] = pd.to_numeric(medication['rel_days'], errors='coerce').astype(int)
    medication = medication[(medication['rel_days'] >= 0)]

    # Pivot to get medications in columns
    meds_pivot = medication.pivot_table(
        index=['patient_id', 'transplant_id', 'rel_days'],
        columns='atc',
        values='ddd',
        aggfunc='sum'
    ).reset_index()

    # for L04AD01 keep only < 2000 mg
    meds_pivot = meds_pivot[~((meds_pivot['L04AD01'] > 2000))]

    # number of patients with L04AD01
    #print("number of patients with L04AD01", meds_pivot['L04AD01'].notna().sum()) # 550

    # for L04AD02 keep only < 500 mg
    meds_pivot = meds_pivot[~((meds_pivot['L04AD02'] > 100))]

    # for L04AA06 keep only < 8000 mg
    meds_pivot = meds_pivot[~((meds_pivot['L04AA06'] > 8000))]

    #print("medication data for patient 2327", meds_pivot[meds_pivot['patient_id'] == '2327'])

    '''
    scalers = {}
    for col in filtered_atcs:
        scaler = MinMaxScaler()
        meds_pivot[col] = scaler.fit_transform(meds_pivot[[col]])
        scalers[col] = scaler'''

    # plot the distribution of the medication
    #import matplotlib.pyplot as plt
    #meds_pivot[filtered_atcs].hist(bins=50, figsize=(20, 15), layout=(3, 2), color='blue', alpha=0.7)
    #plt.suptitle('Distribution of Medication Doses')
    #plt.show()


    return meds_pivot

def create_ts_df(vital_parameters, lab_values, medication, merge_lab=True, merge_med=True):
    ts_data = vital_parameters.copy()

    if merge_lab:
        ts_data = pd.merge(ts_data, lab_values, on=['patient_id', 'transplant_id', 'rel_days'], how='outer')
    
    if merge_med:

        ts_data = pd.merge(
            ts_data,
            medication,
            on=['patient_id', 'transplant_id', 'rel_days'],
            how='left'
        )

    # drop rel_days == nan
    ts_data = ts_data.dropna(subset=['rel_days'])

    # Sorting
    ts_data = ts_data.sort_values(by=['patient_id', 'rel_days']).reset_index(drop=True)

    print("max rel_days", ts_data['rel_days'].max())
    # max number of rows accross all patients
    print("max number of rows across all patients", ts_data.groupby('patient_id').size().max())
    
    return ts_data


def find_index(v, vs, i=0, j=-1):
    if j == -1:
        j = len(vs) - 1

    if v > vs[j]:
        return j + 1
    elif v < vs[i]:
        return i
    elif j - i == 1:
        return j

    k = int((i + j)/2)
    if v <= vs[k]:
        return find_index(v, vs, i, k)
    else:
        return find_index(v, vs, k, j)
    
def generate_feature_dicts(df, feat_list, split_num=100):
    """
    Generate feature statistics and range splits from a DataFrame.
    
    Parameters:
    - df: pandas DataFrame with numeric values (may include NaNs)
    - feat_list: list of feature names (columns in df)
    - split_num: how many bins to split each feature into (default: 100)
    
    Returns:
    - feature_value_dict, feature_range_dict, feature_mm_dict, feature_ms_dict
    """
    feature_value_dict = {}
    feature_range_dict = {}
    feature_mm_dict = {}
    feature_ms_dict = {}

    for feat in feat_list:
        # Drop NaNs, get sorted values
        values = df[feat].dropna().astype(float).sort_values().values.tolist()

        if not values:
            continue

        # Store raw values
        feature_value_dict[feat] = values

        # Generate bin splits
        value_split = []
        for i in range(split_num):
            idx = int(i * len(values) / split_num)
            value_split.append(values[idx])
        value_split.append(values[-1])
        feature_range_dict[feat] = value_split

        # Min/max within trimmed range
        n = int(len(values) / split_num)
        feature_mm_dict[feat] = [values[n], values[-n - 1]]

        # Mean and standard deviation
        feature_ms_dict[feat] = [np.mean(values), np.std(values)]

    return feature_range_dict, feature_mm_dict, feature_ms_dict



class NephroDataset(Dataset):
    def __init__(self, static_df, ts_data, biopsy_df, phase='train'):
        self.phase = phase
        self.static_df = static_df.copy()

        for col in CONFIG['static_numerical_feat']:
            self.static_df[col] = self.static_df[col].fillna(-1)

        # Normalize numerical variables
        self.scaler = StandardScaler()
        self.static_df[CONFIG['static_numerical_feat']] = self.scaler.fit_transform(
            self.static_df[CONFIG['static_numerical_feat']]
        )

        # Encode categorical variables
        self.label_encoders = {}
        for col in CONFIG['static_categorical_feat']:
            le = LabelEncoder()
            self.static_df[col] = le.fit_transform(self.static_df[col])
            self.label_encoders[col] = le

        self.categorical_cardinalities = [len(le.classes_) for le in self.label_encoders.values()]

        self.split_num = CONFIG['split_num']

        # create graft loss labels
        self.labels = self.static_df[['patient_id', 'loss_rel_days', 'death_rel_days', 'risk_label']].copy()
        self.labels['graft_loss_label'] = self.labels['loss_rel_days'].notna().astype(int)
        self.labels['death_label'] = self.labels['death_rel_days'].notna().astype(int)
        self.labels['risk_label'] = self.labels['risk_label'].astype(int)

        self.ts_data = ts_data.copy()
        # value mask for time series data
        self.ts_data_value_mask = self.ts_data[['patient_id', 'transplant_id', 'rel_days']].copy()
        self.ts_data_value_mask[CONFIG['ts_feat']] = (~self.ts_data[CONFIG['ts_feat']].isna()).astype(int)

        # Filter ts_data to patients in static_df
        valid_patient_ids = self.static_df['patient_id'].unique()
        self.ts_data = self.ts_data[self.ts_data['patient_id'].isin(valid_patient_ids)].copy()

        # Keep only patients with enough time series points (e.g., at least 10)
        ts_counts = self.ts_data.groupby('patient_id').size().reset_index(name='counts')
        valid_patient_ids = ts_counts[ts_counts['counts'] >= 10]['patient_id'].unique()

        # Now re-filter all data based on these final valid IDs
        self.static_df = self.static_df[self.static_df['patient_id'].isin(valid_patient_ids)].copy()
        self.labels = self.labels[self.labels['patient_id'].isin(valid_patient_ids)].copy()
        self.ts_data = self.ts_data[self.ts_data['patient_id'].isin(valid_patient_ids)].copy()

        # Generate feature dictionaries
        self.feature_range_dict, self.feature_mm_dict, self.feature_ms_dict = generate_feature_dicts(ts_data, CONFIG['ts_feat'], self.split_num )
        # save feature_mm_dict in jason file
        with open('../feature_ms.json', 'w') as f:
            json.dump(self.feature_ms_dict, f)

        with open('../feature_mm.json', 'w') as f:
            json.dump(self.feature_mm_dict, f)

        # Get unique patient_ids
        self.patient_ids = self.static_df['patient_id'].unique()

        # rejection label for categories 2, 4 
        biopsy_df['Banff 17 categorie'] = pd.to_numeric(biopsy_df['Banff 17 categorie'], errors='coerce')
        rejection_rows = biopsy_df[biopsy_df['Banff 17 categorie'].isin([2, 4])].copy()
        grouped_rejections = (rejection_rows.groupby('PatientID')['BXtime'].apply(list).reset_index(name='rej_rel_days_list'))
        
        # Convert to dictionary: { patient_id: [day1, day2, ...] }
        self.rej_dict = dict(zip(grouped_rejections['PatientID'], grouped_rejections['rej_rel_days_list']))

    def map_input(self, value, feat_list, feat_index):
        # for each feature (index), there are 1 embedding vectors for NA, split_num=100 embedding vectors for different values
        index_start = (feat_index + 1)* (1 + self.split_num) + 1

        if value in ['NA', '']:
            return index_start - 1
        else:
            # print('""' + value + '""')
            value = float(value)
            vs = self.feature_range_dict[feat_list[feat_index]][1:-1]
            v = find_index(value, vs) + index_start
            return v
        
    def map_output(self, value, feat_list, feat_index):
        if value in ['NA', '']:
            return 0
        else:
            value = float(value)
            minv, maxv = self.feature_mm_dict[feat_list[feat_index]]
            if maxv <= minv:
                print(feat_list[feat_index], minv, maxv)
            assert maxv > minv
            v = (value - minv) / (maxv - minv)
            # v = (value - minv) / (maxv - minv)
            v = max(0, min(v, 1))
            return v
        
    def get_pre_info(self, ts_feat: np.ndarray, iline: int, feat_list: list):
        """
        Finds the most recent past non-NaN value and relative time for each feature before iline.
        Calls post_info on reversed window.
        """

        if iline == 0:
            # No past data exists — return zeros
            return [0] * len(feat_list), [0] * len(feat_list)
        
        iline = len(ts_feat) - iline - 1
        reversed_ts_feat = ts_feat[::-1][: -1] # Exclude the current line
        return self.get_post_info(reversed_ts_feat, iline, feat_list)


    def get_post_info(self, ts_feat: np.ndarray, iline: int, feat_list: list):
        """
        Finds the next non-NaN value and relative time for each feature, starting from iline.
        Assumes ts_feat[:, 0] is rel_days and ts_feat[:, 1:] are the features.
        """
        #rel_days = ts_feat[:, 0] # shape: [seq_len]
        #features = ts_feat[:, 1:]  # shape: [seq_len, num_features]
        input_data = ts_feat[iline:]
        #print('input_data:', input_data[:5])
        #print('input_data:', input_data.shape)

        post_input = [0]  # first entry is always 0 (time anchor)
        post_time = [0]

        for feat_index in range(1, len(input_data[0])):
            value = ''
            time_diff = 0
            for j in range(1, len(input_data)):
                val = input_data[j][feat_index]
                #print(type(val))
                if not np.isnan(val):
                    value = val
                    #print('time:', input_data[j][0])
                    time_diff = abs(int(input_data[j][0]) - int(input_data[0][0])) # time difference between current input and the first non-Nan input 
                    break
            mapped_value = self.map_input(value, feat_list, feat_index)
            post_input.append(mapped_value)
            post_time.append(time_diff)

        return post_input, post_time


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # Static Features
        static_row = self.static_df[self.static_df['patient_id'] == patient_id]
        if static_row.empty:
            raise ValueError(f"No static data found for patient_id: {patient_id}")
        static_features = static_row.drop(columns=['patient_id']).iloc[0]

        # Split categorical and numerical features
        categorical_features = torch.tensor(static_features[CONFIG['static_categorical_feat']].values.astype(np.int64))
        numerical_features = torch.tensor(static_features[CONFIG['static_numerical_feat']].values.astype(np.float32))

        # time-series feature tensor
        ts_data = self.ts_data[self.ts_data['patient_id'] == patient_id]
        # just keep the time series features in ts_data
        ts_feat = ts_data[CONFIG['ts_feat_tensor']].values.astype(np.float32)  # Now a NumPy array

        #print(ts_feat[:5])
        #print(ts_feat.shape)

        # save the real ts_feat
        real_ts_feat = ts_feat.copy()

        # count number of nans in each column in ts_real_features
        #num_nans = np.isnan(ts_feat).sum()
        #print('nans before:', num_nans)

        if self.phase == 'train' or 'valid':
            #Create mask for valid entries (non-NaN)
            valid_mask = ~np.isnan(ts_feat)
            # Exclude first column (rel_days) from being masked
            valid_mask[:, 0] = False
            valid_indices = np.argwhere(valid_mask)

            missing_ratio = 0.1

            num_valid = len(valid_indices)
            num_to_mask = int(num_valid * missing_ratio)

            if num_to_mask > 0:
                selected = valid_indices[np.random.permutation(num_valid)[:num_to_mask]]
                for row, col in selected:
                    ts_feat[row, col] = np.nan

        # count number of nans in each column in ts_real_features
        #num_nans = np.isnan(ts_feat).sum()
        #print('nans after:', num_nans)

        # make a copy after adding the nans and before mapping the values, pure values
        init_ts_feat = ts_feat.copy()

        mask_list, input_list, output_list = [], [], []
        pre_input_list, pre_time_list = [], []
        post_input_list, post_time_list = [], []

        seq_len, num_features = ts_feat.shape
        #print('seq_len:', seq_len, 'num_features:', num_features)
        #print('loss rel days:', int(static_row['loss_rel_days']))
        for iline in range(seq_len):
            #ctime = int(rel_days[iline])  # current time step

            ctime = int(ts_feat[iline, 0])  # current time step
            #diff = abs(ctime - int(static_row['loss_rel_days']))
            #rel_diff_list.append(diff)
    
            # including time
            input_row = ts_feat[iline]
            output_row = real_ts_feat[iline]
            init_row = init_ts_feat[iline]  # assuming original is used as reference

            #print('shape input_row:', input_row.shape)
            mask = []
            input_vec = []
            output_vec = []

            for i in range(1, num_features):
                iv = input_row[i]
                ov = output_row[i]
                init_iv = init_row[i]
                #print(type(iv))

                # masking logic
                if not np.isnan(init_iv):
                    mask.append(0)  # observed
                elif not np.isnan(ov):
                    mask.append(1)  # can be predicted
                else:
                    mask.append(-1)  # completely missing

                input_vec.append(self.map_input(iv, CONFIG['ts_feat_tensor'], i))
                output_vec.append(self.map_output(ov, CONFIG['ts_feat_tensor'], i))

            mask_list.append(mask)
            input_list.append(input_vec)
            output_list.append(output_vec)

            #print(init_ts_feat.shape)

            pre_input, pre_time = self.get_pre_info(init_ts_feat, iline, CONFIG['ts_feat_tensor'])
            pre_input_list.append(pre_input)
            pre_time_list.append(pre_time)

            post_input, post_time = self.get_post_info(init_ts_feat, iline, CONFIG['ts_feat_tensor'])
            post_input_list.append(post_input)
            post_time_list.append(post_time)


        #print(real_ts_data.head())
        #print('mask_list:', mask_list[:5])
        #print('input_list:', input_list[:5])
        #print('output_list:', output_list[:5])
        #print('pre_input_list:', pre_input_list[:5])
        #print('pre_time_list:', pre_time_list[:5])

        #pre_time_tensor = torch.tensor([[int(round(t)) for t in row] for row in pre_time_list], dtype=torch.int64)
        #post_time_tensor = torch.tensor([[int(round(t)) for t in row] for row in post_time_list], dtype=torch.int64)
        #input_list = input_list[:, 1:]
        #output_list = output_list[:, 1:]
        #mask_list = mask_list[:, 1:]
        pre_time_list = np.array(pre_time_list, dtype=np.int64)
        post_time_list = np.array(post_time_list, dtype=np.int64)
        pre_time_list[pre_time_list>200] = 200
        post_time_list[post_time_list>200] = 200

        # Convert to tensor if needed
        #rel_diff_tensor = torch.tensor(rel_diff_list, dtype=torch.int64)
        pre_input_tensor = torch.tensor(pre_input_list, dtype=torch.int64)
        pre_time_tensor = torch.tensor(pre_time_list, dtype=torch.int64)
        post_input_tensor = torch.tensor(post_input_list, dtype=torch.int64)
        post_time_tensor = torch.tensor(post_time_list, dtype=torch.int64)

        mask_tensor = torch.tensor(mask_list, dtype=torch.int8)

        input_ts_data = torch.tensor(input_list, dtype=torch.int64)
        output_ts_data = torch.tensor(output_list, dtype=torch.float32)

        init_ts_data = torch.tensor(init_ts_feat, dtype=torch.float32) # real ts data with time 

        #print('input_ts_data:', input_ts_data.shape)
        #print('output_ts_data:', output_ts_data.shape)
        #print('mask_tensor:', mask_tensor.shape)
        #print('pre_input_tensor:', pre_input_tensor.shape)
        #print('pre_time_tensor:', pre_time_tensor.shape)
        #print('post_input_tensor:', post_input_tensor.shape)
        #print('post_time_tensor:', post_time_tensor.shape)
        #print('real_ts_data:', init_ts_data.shape)

        # check if the pre and post input and time tensors have nans
        if np.isnan(pre_input_tensor).any():
            print('pre_input_tensor has NaNs')
        if np.isnan(pre_time_tensor).any():
            print('pre_time_tensor has NaNs')
        if np.isnan(post_input_tensor).any():
            print('post_input_tensor has NaNs')
        if np.isnan(post_time_tensor).any():
            print('post_time_tensor has NaNs')

        seq_len = input_ts_data.shape[0]
        # id seq len is nan print the seq len
        if np.isnan(seq_len):
            print('seq_len is NaN')
            print('patient_id:', patient_id)

        # get labels
        labels = self.labels[self.labels['patient_id'] == patient_id]
        graft_loss_label = torch.tensor(labels['graft_loss_label'].values.astype(np.int64))
        death_label = torch.tensor(labels['death_label'].values.astype(np.int64))
        risk_label = torch.tensor(labels['risk_label'].values.astype(np.int64))
        loss_rel_days = torch.tensor(static_row['loss_rel_days'].values.astype(np.float32))
        death_rel_days = torch.tensor(static_row['death_rel_days'].values.astype(np.float32))

        # rejection labels
        rej_rel_days_list = self.rej_dict.get(patient_id, [])
        if len(rej_rel_days_list) == 0:
            rej_rel_days_list = None

        #print('mask shape:',  mask_tensor.shape) # shape: [seq_len, num_features]
        #print('input_ts_data shape:', input_ts_data.shape) # shape: [seq_len, num_features]
        #print('pre_input_tensor shape:', pre_input_tensor.shape) # shape: [seq_len, num_features]
        #print('output_ts_data shape:', output_ts_data.shape) # shape: [seq_len, num_features]
        #print('categorical_features shape:', categorical_features.shape) # shape: [num_categorical_features]
        #print('numerical_features shape:', numerical_features.shape) # shape: [num_numerical_features]


        sample = {
            'patient_id': patient_id,
            'static_numerical_features': numerical_features,
            'static_categorical_features': categorical_features,
            'input_ts_features': input_ts_data,
            'output_ts_features': output_ts_data,
            'pre_input': pre_input_tensor,
            'pre_time': pre_time_tensor,
            'post_input': post_input_tensor,
            'post_time': post_time_tensor,
            'value_mask': mask_tensor,
            'seq_len': seq_len,
            'loss_rel_days': loss_rel_days,
            'risk_label': risk_label,
            'graft_loss_label': graft_loss_label,
            'death_label': death_label,
            'death_rel_days': death_rel_days,
            'rej_rel_days_list': rej_rel_days_list, # List of rejection days or None
            'real_ts_data': init_ts_data # Real time series data with relative days
        }

        return sample
    

def collate_fn(batch):
    print("Collecting batch...")
    collated_batch = {}
    global_max_seq_len = CONFIG['global_max_seq_len']

    for key in batch[0].keys():
        data = [item[key] for item in batch]

        if key in ['input_ts_features','output_ts_features','pre_input', 'pre_time', 'post_input', 'post_time', 'value_mask', 'real_ts_data']:
            padding_value = 0 if key != 'value_mask' else -1
            padded = pad_sequence(data, batch_first=True, padding_value=padding_value)
            
            # Pad or truncate to fixed global length
            if padded.size(1) < global_max_seq_len:
                pad_len = global_max_seq_len - padded.size(1)
                pad_tensor = torch.full((padded.size(0), pad_len, *padded.shape[2:]), padding_value, dtype=padded.dtype)
                padded = torch.cat([padded, pad_tensor], dim=1)
            elif padded.size(1) > global_max_seq_len:
                padded = padded[:, :global_max_seq_len, ...]

            collated_batch[key] = padded

        elif key == 'seq_len':
            seq_lengths = torch.tensor(data, dtype=torch.long)
            collated_batch[key] = seq_lengths

        elif all(isinstance(d, torch.Tensor) and d.shape == data[0].shape for d in data):
            collated_batch[key] = torch.stack(data)

        else:
            collated_batch[key] = data

    # Also fix the generated mask
    batch_size = len(batch)
    mask = torch.arange(global_max_seq_len).expand(batch_size, global_max_seq_len) < collated_batch['seq_len'].unsqueeze(1)
    collated_batch['mask'] = mask

    return collated_batch



    
