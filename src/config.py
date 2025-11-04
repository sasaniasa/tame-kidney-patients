CONFIG = {
    'static_categorical_feat': ['underlying_disease', 'blood_group', 'ebv_status', 'delayed_graft_function', 'donor_bloodgroup', \
                                'type_of_donation', 'gender', 'gender_donor'
                               ], # ,'death_label', 'loss_label'

    'static_numerical_feat':  ['number_dialyses', 'age', 'height', 'cold_ischemia_time','age_donor', 'days_in_hospital', \
                               'number_transplantation','pirche_score', 'risk_of_graft_failure'],


    'ts_feat': ['bp_sys', 'bp_dia', 'weight', 'urine_volume', 'heart_rate', 'temperature', 'diuresis_time', 'albumin', 'crphp', 
                'creatinine', 'leukocyte', 'proteinuria', 'L04AA06', 'L04AD01','L04AD02'],
    
    'ts_feat_tensor': ['rel_days', 'bp_sys', 'bp_dia', 'weight', 'urine_volume', 'heart_rate', 'temperature', 'diuresis_time', 'albumin', 'crphp', 
                'creatinine', 'leukocyte', 'proteinuria', 'L04AA06', 'L04AD01','L04AD02'],


    'split_num': 200,
    'embedding_dim': 256,
    'hidden_dim': 256,
    'max_rel_days': 8034,
    'global_max_seq_len': 200, # longer length introduce bias
    'batch_size': 32
}
