"""
Retrain all ML models from recotrip_dataset.csv (the enhanced dataset).
This is needed because the .pkl files were saved with scikit-learn 1.6.1
but we are now running 1.9.0, which causes InconsistentVersionWarning.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib

print("Step 1: Loading enhanced dataset...")
df_main = pd.read_csv('recotrip_dataset.csv')
print(f"Loaded {len(df_main)} places")

# Ensure required columns exist (they should already be in the enhanced CSV)
required = ['Ideal_Season', 'Age_Group_Suitability', 'Budget_Category', 'time needed to visit in hrs']
for col in required:
    if col not in df_main.columns:
        raise ValueError(f"Missing column: {col}")

print("\nStep 2: Encoding features...")

encoders = {}
encoded_features = pd.DataFrame()

for col in ['Type', 'Significance', 'State', 'Zone', 'Budget_Category']:
    le = LabelEncoder()
    encoded_features[col] = le.fit_transform(df_main[col].astype(str))
    encoders[col] = le

mlb_season = MultiLabelBinarizer()
season_encoded = mlb_season.fit_transform(df_main['Ideal_Season'].str.split(','))

mlb_age = MultiLabelBinarizer()
age_encoded = mlb_age.fit_transform(df_main['Age_Group_Suitability'].str.split(','))

X_combined = np.hstack([
    encoded_features.values,
    season_encoded,
    age_encoded,
    df_main[['Google review rating']].values,
    df_main[['time needed to visit in hrs']].values
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

print(f"Feature matrix shape: {X_scaled.shape}")

print("\nStep 3: Training K-Means (k=15)...")
kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
df_main['Cluster'] = kmeans.fit_predict(X_scaled)
print(f"Clusters formed: {df_main['Cluster'].nunique()}")
print(f"Cluster sizes:\n{df_main['Cluster'].value_counts().sort_index()}")

print("\nStep 4: Saving models and data...")
joblib.dump(kmeans,        'recotrip_kmeans_model.pkl')
joblib.dump(encoders,      'recotrip_encoders.pkl')
joblib.dump(mlb_season,    'recotrip_season_encoder.pkl')
joblib.dump(mlb_age,       'recotrip_age_encoder.pkl')
joblib.dump(scaler,        'recotrip_scaler.pkl')
df_main.to_csv('recotrip_places_with_clusters.csv', index=False)

print("Saved: recotrip_kmeans_model.pkl")
print("Saved: recotrip_encoders.pkl")
print("Saved: recotrip_season_encoder.pkl")
print("Saved: recotrip_age_encoder.pkl")
print("Saved: recotrip_scaler.pkl")
print("Saved: recotrip_places_with_clusters.csv")
print("\nAll models retrained on sklearn", end=" ")
import sklearn
print(sklearn.__version__)
print("Done!")
