import pandas as pd, numpy as np, json, pickle, os
from ml_model import engineer_features, FEATURE_COLS
from rule_engine import run_rule_engine
from explainer import explain_transaction

df_raw = pd.read_csv('transactions.csv', parse_dates=['timestamp'])
df = engineer_features(df_raw)
df_rules = run_rule_engine(df_raw.copy())

with open('fraud_model.pkl','rb') as f:
    art = pickle.load(f)
X = df[FEATURE_COLS].values
X_scaled = art['scaler'].transform(X)
df['ml_prediction'] = art['model'].predict(X_scaled)
df['ml_probability'] = np.round(art['model'].predict_proba(X_scaled)[:,1], 4)
df['rule_burst'] = df_rules['rule_burst'].values
df['rule_geo_mismatch'] = df_rules['rule_geo_mismatch'].values
df['rule_odd_hour'] = df_rules['rule_odd_hour'].values
df['risk_score'] = df_rules['risk_score'].values
df['combined_prediction'] = ((df['ml_prediction']==1)|(df['risk_score']>0)).astype(int)

total = len(df)
fraud = int(df['is_fraud'].sum())
summary = {
    'total_transactions': total,
    'actual_fraud': fraud,
    'actual_normal': total - fraud,
    'flagged_by_ml': int(df['ml_prediction'].sum()),
    'flagged_by_rules': int((df['risk_score']>0).sum()),
    'flagged_combined': int(df['combined_prediction'].sum()),
    'model_metrics': {'precision': 0.800, 'recall': 0.774, 'f1': 0.787, 'auc_roc': 0.980}
}

bins = list(np.arange(0, 1.05, 0.05))
hist_fraud, _ = np.histogram(df[df['is_fraud']==1]['ml_probability'], bins=bins)
hist_normal, _ = np.histogram(df[df['is_fraud']==0]['ml_probability'], bins=bins)
risk_dist = {
    'labels': [f'{b:.2f}-{b+0.05:.2f}' for b in bins[:-1]],
    'fraud': hist_fraud.tolist(),
    'normal': hist_normal.tolist()
}

df_ts = df.copy()
df_ts['date'] = df_ts['timestamp'].dt.date.astype(str)
daily = df_ts.groupby(['date','is_fraud']).size().unstack(fill_value=0).reset_index()
daily.columns = ['date','normal','fraud'] if 0 in daily.columns and 1 in daily.columns else ['date']
if 'normal' not in daily.columns: daily['normal']=0
if 'fraud' not in daily.columns: daily['fraud']=0
time_series = {'dates': daily['date'].tolist(), 'normal': daily['normal'].tolist(), 'fraud': daily['fraud'].tolist()}

flagged = df[df['combined_prediction']==1].copy()
cols = ['transaction_id','amount','timestamp','card_id','device_id','merchant_id','geo_location','is_fraud','ml_prediction','ml_probability','rule_burst','rule_geo_mismatch','rule_odd_hour','risk_score']
flagged_out = flagged[cols].head(200).copy()
flagged_out['timestamp'] = flagged_out['timestamp'].astype(str)
flagged_out['ml_probability'] = flagged_out['ml_probability'].round(4)
flagged_list = flagged_out.to_dict(orient='records')

explanations = {}
for t in flagged_list[:50]:
    explanations[t['transaction_id']] = explain_transaction(t)

data = {
    'summary': summary,
    'risk_distribution': risk_dist,
    'time_series': time_series,
    'flagged': flagged_list,
    'explanations': explanations
}

os.makedirs('docs', exist_ok=True)
with open('docs/data.json', 'w') as f:
    json.dump(data, f)

fraud_dist = risk_dist['fraud']
normal_dist = risk_dist['normal']
print(f"Generated docs/data.json")
print(f"Flagged: {len(flagged_list)}, Explanations: {len(explanations)}")
print(f"Risk distribution fraud:    {fraud_dist}")
print(f"Risk distribution normal:   {normal_dist}")
