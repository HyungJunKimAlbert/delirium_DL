import pandas as pd
import numpy as np
import os, tqdm, pickle
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)  # convert exponential to integer

from utils.util import qSOFA_Score, get_label




if __name__ == "__main__":
# 01. Set Options
    data_use = 24 * 60
    # vitals
    vital_list = ['Heart Rate', 'Respiratory Rate', 'O2 Saturation', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Systolic', 'Temperature (C)']
    # vital_list = ['Heart Rate']
    # final labs  =>>> MCH / RDW / albumin / AST / ALT / total protein / total bilirubin / PaO2 / PaCO2 / pH / PT-INR / HCO3 / FiO2 / PTT / creatinine / RBC 제외
    lab_list = ['sodium', 'glucose', 'Hgb', 'chloride', 'Hct', 'BUN', 'calcium', 'bicarbonate', 'platelets x 1000', 'WBC x 1000', 'MCV', 'MCHC']
    disease_list = ['angina', 'mi', 'stroke', 'afib', 'dementia']
    medication_list = ['fentanyl', 'morphine', 'lorazepam','ativan','midazolam','furosemide','metoprolol',
                    'propofol','aspirin','mycin']

# 02. Import Dataset
    data_path = "/home/hjkim/projects/local_dev/delirium/features"
    demo = pd.read_feather(os.path.join(data_path, "demographics_cleansed.ftr"))
    vitals = pd.read_feather(os.path.join(data_path, "vitals_cleansed.ftr")).astype({'nursingchartvalue':'float'})   # [['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue']]
    labs = pd.read_feather(os.path.join(data_path, "labs_cleansed.ftr")).astype({'labresult':'float'})
    scores = pd.read_feather(os.path.join(data_path, "scores_cleansed.ftr"))
    ventilator = pd.read_feather(os.path.join(data_path, "ventilator_cleansed.ftr"))
    rass = pd.read_feather(os.path.join(data_path, "rass_cleansed.ftr"))
    history = pd.read_feather(os.path.join(data_path, "history_cleansed.ftr"))
    drugs = pd.read_feather(os.path.join(data_path, "drugs_cleansed.ftr"))

    final_label = get_label(demo)
    print('TOTAL: DATASET: ', len(final_label))     
    for lead_time in [2*60, 3*60, 1*60]:

    # 03. Create Features
        data = dict()
        data['vitals'], data['emr'], data['labs'], data['medication'], data['label'] = [], [], [], [], []
        # data['emr'] = []

        # Extract all features by patient stay_id 
        for idx, value in tqdm.tqdm(enumerate(final_label.iterrows())):
            stay_id = final_label.patientunitstayid[idx]
            upper_bound = final_label.charttime[idx] - lead_time
            lower_bound = upper_bound - data_use
            label = final_label.delirium[idx]
            los = upper_bound // 60  # Hours

            # vitals (time-series)
            vitals_tmp = vitals[(vitals.patientunitstayid==stay_id) & (vitals.nursingchartoffset < upper_bound) & (vitals.nursingchartoffset > lower_bound)]
            if len(vitals_tmp) == 0:
                continue
            # Loss over than 50%
            if len(vitals_tmp.drop_duplicates(['nursingchartoffset'])) < (data_use / 5) * 0.15 :  #  (data_use / 5) * 0.5    
                continue
            
            # Demographics (idx: 1-gender, 2-age, 3-height, 4-weight, 5-apachescore)
            emr_result = [los]
            demo_tmp = demo[demo.patientunitstayid==stay_id].iloc[0, 1:3].tolist() + [demo[demo.patientunitstayid==stay_id].iloc[0, 5]]
            emr_result.extend(demo_tmp)

            # Ventilator
            ventil_tmp = ventilator[(ventilator.patientunitstayid==stay_id) & (labs.labresultoffset < upper_bound)]
            if len(ventil_tmp) > 0:
                ventil_result = 1
            else:
                ventil_result = 0
            emr_result.append(ventil_result)
            # GCS
            gcs_tmp = scores[(scores.patientunitstayid==stay_id) & (scores.nursingchartoffset < upper_bound) & (scores.nursingchartoffset > lower_bound)].sort_values(['nursingchartoffset'], ascending=False)
            gcs_data = gcs_tmp.nursingchartvalue.mean() if not gcs_tmp.empty else -1
            gcs_latest = gcs_tmp.iloc[0].nursingchartvalue if not gcs_tmp.empty else -1
            emr_result.append(gcs_data)
            emr_result.append(gcs_latest)
            # RASS
            rass_tmp = rass[(rass.patientunitstayid==stay_id) & (rass.nursingchartoffset < upper_bound) & (rass.nursingchartoffset > lower_bound)].sort_values(by=['nursingchartoffset'], ascending=False)
            rass_data = rass_tmp.nursingchartvalue.mean() if not rass_tmp.empty else -10
            rass_latest = rass_tmp.iloc[0].nursingchartvalue if not rass_tmp.empty else -10
            emr_result.append(rass_data)
            emr_result.append(rass_latest)

            # History [angina, mi, dementia, atrial fibrillation%, stroke]
            hist_list = [0] * len(disease_list)
            history_tmp = history[(history.patientunitstayid==stay_id) & (history.pasthistoryoffset < upper_bound)]
            for idx in range(len(disease_list)):
                if len(history_tmp[history_tmp.category==disease_list[idx]]) > 0:
                    hist_list[idx] = 1
            emr_result.extend(hist_list)

            # LAB           
            lab_result = []
            labs_tmp = labs[(labs.patientunitstayid==stay_id) & (labs.labresultoffset < upper_bound) & (labs.labresultoffset > lower_bound)].sort_values(['labresultoffset'], ascending=False)
            for l in lab_list:
                if len(labs_tmp) == 0:
                    lab_data, lab_latest = -1, -1
                else:
                    lab_data = labs_tmp[labs_tmp.labname==l].labresult.mean()
                    lab_latest = labs_tmp.iloc[0].labresult
                lab_result.append(lab_data)
                lab_result.append(lab_latest)

            # Medication
            drug_result = []
            drug_list = [0] * len(medication_list)

            drug_tmp = drugs[(drugs.patientunitstayid==stay_id) & (drugs.drugstartoffset < upper_bound) & (drugs.drugstartoffset > lower_bound)]
            for idx in range(len(medication_list)):
                if len(drug_tmp[drug_tmp.category==medication_list[idx]]) > 0:
                    drug_list[idx] = 1
            drug_result.extend(drug_list)

            # Vital sign
            vital_result = pd.DataFrame()
            for v in vital_list:
                data_df = vitals_tmp[vitals_tmp.nursingchartcelltypevalname==v][["nursingchartoffset", "nursingchartvalue"]].rename(columns={"nursingchartvalue":v})
                bins = pd.cut(data_df['nursingchartoffset'], bins=range(lower_bound, upper_bound+1, 5))
                data_df2 = pd.DataFrame(data_df.groupby(bins).mean())[[v]].reset_index().drop(columns=['nursingchartoffset'])
                vital_result = pd.concat([vital_result, data_df2], axis=1).ffill().fillna(-1)
            
            # Vital statistics in EMR result
            vital_statistics = vital_result.iloc[-96:,:]   # 96=8Hours
            emr_result.extend(list(vital_result.replace(-1, np.nan).min(axis=0)))
            emr_result.extend(list(vital_result.replace(-1, np.nan).max(axis=0)))
            emr_result.extend(list(vital_result.replace(-1, np.nan).median(axis=0)))
            emr_result.extend(list(vital_result.replace(-1, np.nan).mean(axis=0)))
            emr_result.extend(list(vital_result.replace(-1, np.nan).std(axis=0)))

            # qSOFA
            sys_bp = vital_result['Non-Invasive BP Diastolic'].iloc[-1]
            rr = vital_result['Respiratory Rate'].iloc[-1]
            qsofa_score = qSOFA_Score(sys_bp, rr, gcs_latest)
            emr_result.append(qsofa_score)

            # Add EMR data
            data['emr'].append(np.array(emr_result))
            # Add Lab data
            data['labs'].append(np.array(lab_result))
            # Add vitals data
            data['vitals'].append(np.array(vital_result))
            # Add label data
            data['label'].append(label)
            data['medication'].append(drug_result)

            if idx % 1000 == 0:
                with open('data_dynamic_' + str(lead_time//60) + 'h_add_total.pkl','wb') as fw:
                    pickle.dump(data, fw)
                    
        print("LABEL: ", np.array(data['label']).shape)
        print("EMR(DEMOGRAPHICS+ETC): ", np.array(data['emr']).shape)
        print("LABS: ", np.array(data['labs']).shape)
        print("MEDICATION: ", np.array(data['medication']).shape)
        print("VITALS: ", np.array(data['vitals']).shape)

    # 04. Save Results
        # save data
        with open('data_dynamic_' + str(lead_time//60) + 'h_add_total.pkl','wb') as fw:
            pickle.dump(data, fw)
