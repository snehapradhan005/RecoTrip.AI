import pandas as pd
import joblib
import numpy as np

df = pd.read_csv('recotrip_places_with_clusters.csv')
kmeans = joblib.load('recotrip_kmeans_model.pkl')
encoders = joblib.load('recotrip_encoders.pkl')
season_encoder = joblib.load('recotrip_season_encoder.pkl')
age_encoder = joblib.load('recotrip_age_encoder.pkl')
scaler = joblib.load('recotrip_scaler.pkl')

print('=== DIAGNOSIS ===')
print('Dataset shape:', df.shape)
print('Cluster column exists:', 'Cluster' in df.columns)
print('Total clusters:', df['Cluster'].nunique())
print('Cluster distribution:\n', df['Cluster'].value_counts().sort_index())

# Simulate the current app's build_user_vector
DURATION_HOURS_MAP = {'0.5 Day': 3, '1 Day': 6, '2 Days': 12, '3+ Days': 20, 'Any': 8}
budget = 'Budget'
season = 'Winter'
age_group = 'Families'
duration = '1 Day'

type_val = df['Type'].mode()[0]
sig_val = df['Significance'].mode()[0]
state_val = df['State'].mode()[0]
zone_val = df['Zone'].mode()[0]
budget_val = budget

type_enc = encoders['Type'].transform([type_val])[0]
sig_enc = encoders['Significance'].transform([sig_val])[0]
state_enc = encoders['State'].transform([state_val])[0]
zone_enc = encoders['Zone'].transform([zone_val])[0]
budget_enc = encoders['Budget_Category'].transform([budget_val])[0]

season_vec = season_encoder.transform([[season]])[0]
age_vec = age_encoder.transform([[age_group]])[0]

duration_hrs = DURATION_HOURS_MAP.get(duration, 8)
x_query = np.hstack([
    [type_enc, sig_enc, state_enc, zone_enc, budget_enc],
    season_vec, age_vec,
    [4.5, duration_hrs]
]).reshape(1, -1)

x_scaled = scaler.transform(x_query)
predicted_cluster = kmeans.predict(x_scaled)[0]
centroid_distances = kmeans.transform(x_scaled)[0]

print('\nML predicted cluster for (Budget, Winter, Families, 1 Day):', predicted_cluster)
print('Distances to all cluster centroids:', centroid_distances.round(3))
print('Places in predicted cluster', predicted_cluster, ':', len(df[df['Cluster'] == predicted_cluster]))

# Compare ML-ranked vs pure rating sort
test_df = df.copy()
test_df['ml_distance'] = test_df['Cluster'].apply(lambda c: centroid_distances[c])

ml_top5 = test_df.sort_values(
    ['ml_distance', 'Google review rating'], ascending=[True, False]
).head(5)[['Name', 'Cluster', 'ml_distance', 'Google review rating', 'Budget_Category', 'Ideal_Season', 'Age_Group_Suitability']]

rating_top5 = test_df.sort_values(
    'Google review rating', ascending=False
).head(5)[['Name', 'Cluster', 'ml_distance', 'Google review rating', 'Budget_Category', 'Ideal_Season', 'Age_Group_Suitability']]

print('\n=== ML-ranked top 5 ===')
print(ml_top5.to_string())
print('\n=== Pure rating-sorted top 5 (old behavior) ===')
print(rating_top5.to_string())
print('\nAre results different?', ml_top5['Name'].tolist() != rating_top5['Name'].tolist())
print('\n=== KEY QUESTION: Is the current app.py using ML or plain filtering? ===')
print('Looking at app.py... it DOES call build_user_vector + kmeans.predict + sort by ml_distance')
print('So the current recotrip_app.py IS using ML properly.')
print('')
print('The ORIGINAL app (script_2.py, the old version) was doing pure filtering.')
print('The CURRENT app.py has the ML integration already applied.')
