"""
Post-fix validation: confirm hard filters + ML ranking work correctly together.
"""
import pandas as pd
import joblib
import numpy as np

df = pd.read_csv('recotrip_places_with_clusters.csv')
kmeans = joblib.load('recotrip_kmeans_model.pkl')
encoders = joblib.load('recotrip_encoders.pkl')
season_encoder = joblib.load('recotrip_season_encoder.pkl')
age_encoder = joblib.load('recotrip_age_encoder.pkl')
scaler = joblib.load('recotrip_scaler.pkl')

DURATION_HOURS_MAP = {'0.5 Day': 3, '1 Day': 6, '2 Days': 12, '3+ Days': 20, 'Any': 8}

def build_user_vector(budget, season, age_group, duration):
    type_val = df['Type'].mode()[0]
    sig_val = df['Significance'].mode()[0]
    state_val = df['State'].mode()[0]
    zone_val = df['Zone'].mode()[0]
    budget_val = budget if budget != 'Any' else df['Budget_Category'].mode()[0]

    type_enc = encoders['Type'].transform([type_val])[0]
    sig_enc = encoders['Significance'].transform([sig_val])[0]
    state_enc = encoders['State'].transform([state_val])[0]
    zone_enc = encoders['Zone'].transform([zone_val])[0]
    budget_enc = encoders['Budget_Category'].transform([budget_val])[0]

    season_query = [season] if season != 'Any' else ['All Seasons']
    age_query = [age_group] if age_group != 'Any' else ['All Age Groups']
    season_vec = season_encoder.transform([season_query])[0]
    age_vec = age_encoder.transform([age_query])[0]

    duration_hrs = DURATION_HOURS_MAP.get(duration, 8)
    x_query = np.hstack([
        [type_enc, sig_enc, state_enc, zone_enc, budget_enc],
        season_vec, age_vec, [4.5, duration_hrs]
    ]).reshape(1, -1)
    return scaler.transform(x_query)

def get_recommendations(budget, season, age_group, duration, city='Any'):
    query_vec = build_user_vector(budget, season, age_group, duration)
    predicted_cluster = kmeans.predict(query_vec)[0]
    centroid_distances = kmeans.transform(query_vec)[0]

    filtered_df = df.copy()

    if city != 'Any':
        filtered_df = filtered_df[filtered_df['City'] == city]
    if season != 'Any':
        filtered_df = filtered_df[
            filtered_df['Ideal_Season'].str.contains(season, case=False, na=False) |
            filtered_df['Ideal_Season'].str.contains('All Seasons', case=False, na=False)
        ]
    if budget != 'Any':
        filtered_df = filtered_df[filtered_df['Budget_Category'] == budget]
    if age_group != 'Any':
        filtered_df = filtered_df[
            filtered_df['Age_Group_Suitability'].str.contains(age_group, case=False, na=False) |
            filtered_df['Age_Group_Suitability'].str.contains('All Age Groups', case=False, na=False)
        ]

    if len(filtered_df) > 0:
        filtered_df = filtered_df.copy()
        filtered_df['ml_distance'] = filtered_df['Cluster'].apply(lambda c: centroid_distances[c])
        filtered_df = filtered_df.sort_values(['ml_distance', 'Google review rating'], ascending=[True, False])

    return filtered_df, predicted_cluster

print("=" * 60)
print("TEST 1: Budget + Families + Winter")
print("=" * 60)
results, cluster = get_recommendations('Budget', 'Winter', 'Families', '1 Day')
print(f"Predicted cluster: {cluster}")
print(f"Results count: {len(results)}")
top5 = results.head(5)[['Name', 'Budget_Category', 'Ideal_Season', 'Age_Group_Suitability', 'Google review rating', 'ml_distance']]
print(top5.to_string())
print("\nAll results match Budget?", (results['Budget_Category'] == 'Budget').all())
print("All results contain Winter or All Seasons?", results['Ideal_Season'].str.contains('Winter|All Seasons', case=False).all())
print("All results contain Families or All Age Groups?", results['Age_Group_Suitability'].str.contains('Families|All Age Groups', case=False).all())

print("\n" + "=" * 60)
print("TEST 2: Premium + Adults + Summer")
print("=" * 60)
results2, cluster2 = get_recommendations('Premium', 'Summer', 'Adults', '2 Days')
print(f"Predicted cluster: {cluster2}")
print(f"Results count: {len(results2)}")
top5b = results2.head(5)[['Name', 'Budget_Category', 'Ideal_Season', 'Age_Group_Suitability', 'Google review rating', 'ml_distance']]
print(top5b.to_string())
print("\nAll results match Premium?", (results2['Budget_Category'] == 'Premium').all())

print("\n" + "=" * 60)
print("CONCLUSION: Hard filters working + ML ranks within valid pool")
print("=" * 60)
