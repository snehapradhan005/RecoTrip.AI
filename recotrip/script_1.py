
# First, let's recreate all the essential files from scratch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib

# Load the original data
df_main = pd.read_csv('Top-Indian-Places-to-Visit.csv')

print("Step 1: Loading original dataset...")
print(f"✓ Loaded {len(df_main)} places")

# Helper functions
def suggest_season(state, city):
    northern_states = ['Delhi', 'Punjab', 'Haryana', 'Uttarakhand', 'Himachal Pradesh', 'Jammu and Kashmir']
    coastal_states = ['Goa', 'Kerala', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'West Bengal']
    hill_stations = ['Shimla', 'Manali', 'Ooty', 'Darjeeling', 'Munnar']
    
    if state in northern_states:
        return 'Winter,Spring'
    elif state in coastal_states:
        return 'Winter,Spring,Summer'
    elif city in hill_stations:
        return 'Summer,Autumn'
    else:
        return 'All Seasons'

def suggest_age_group(place_type, significance):
    family_types = ['Theme Park', 'Amusement Park', 'Zoo', 'Park', 'Beach', 'Botanical Garden']
    adventure_types = ['National Park', 'Wildlife Sanctuary', 'Trekking', 'Adventure Sport', 'Waterfall']
    cultural_types = ['Temple', 'Fort', 'Palace', 'Museum', 'Monument', 'Historical']
    
    if place_type in family_types:
        return 'Families,Kids,Adults'
    elif place_type in adventure_types:
        return 'Young Adults,Adults'
    elif place_type in cultural_types:
        return 'All Age Groups'
    elif significance == 'Religious':
        return 'All Age Groups'
    else:
        return 'Adults,Young Adults'

def calculate_budget(entrance_fee):
    if pd.isna(entrance_fee) or entrance_fee == 0:
        return 'Budget'
    elif entrance_fee < 50:
        return 'Budget'
    elif entrance_fee < 200:
        return 'Mid-Range'
    else:
        return 'Premium'

print("\nStep 2: Enhancing dataset with new features...")

# Add new columns
df_main['Ideal_Season'] = df_main.apply(lambda x: suggest_season(x['State'], x['City']), axis=1)
df_main['Age_Group_Suitability'] = df_main.apply(lambda x: suggest_age_group(x['Type'], x['Significance']), axis=1)
df_main['Budget_Category'] = df_main['Entrance Fee in INR'].apply(calculate_budget)
df_main['Recommended_Duration_Days'] = df_main['time needed to visit in hrs'].apply(
    lambda x: '0.5 Day' if x <= 4 else ('1 Day' if x <= 8 else '2 Days')
)
df_main['Ideal_Group_Size'] = df_main['Type'].apply(
    lambda x: 'Any' if x in ['Temple', 'Museum', 'Fort', 'Palace'] else 
              ('2-4 people' if x in ['Theme Park', 'Amusement Park'] else 
               ('4+ people' if x in ['National Park', 'Beach'] else 'Any'))
)

print("✓ Added 5 new feature columns")

# Save enhanced dataset
df_main.to_csv('recotrip_dataset.csv', index=False)
print("✓ Saved: recotrip_dataset.csv")

print("\nStep 3: Training Machine Learning models...")

# Prepare features for clustering
features_for_clustering = df_main[['Type', 'Significance', 'State', 'Zone', 'Budget_Category']].copy()

encoders = {}
encoded_features = pd.DataFrame()

for col in ['Type', 'Significance', 'State', 'Zone', 'Budget_Category']:
    le = LabelEncoder()
    encoded_features[col] = le.fit_transform(features_for_clustering[col].astype(str))
    encoders[col] = le

# Multi-label encoders
mlb_season = MultiLabelBinarizer()
season_encoded = mlb_season.fit_transform(df_main['Ideal_Season'].str.split(','))

mlb_age = MultiLabelBinarizer()
age_encoded = mlb_age.fit_transform(df_main['Age_Group_Suitability'].str.split(','))

# Combine features
X_combined = np.hstack([
    encoded_features.values,
    season_encoded,
    age_encoded,
    df_main[['Google review rating']].values,
    df_main[['time needed to visit in hrs']].values
])

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Train K-Means
kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
df_main['Cluster'] = kmeans.fit_predict(X_scaled)

print("✓ Trained K-Means model with 15 clusters")

# Save all models
joblib.dump(kmeans, 'recotrip_kmeans_model.pkl')
joblib.dump(encoders, 'recotrip_encoders.pkl')
joblib.dump(mlb_season, 'recotrip_season_encoder.pkl')
joblib.dump(mlb_age, 'recotrip_age_encoder.pkl')
joblib.dump(scaler, 'recotrip_scaler.pkl')

print("✓ Saved all 5 model files (.pkl)")

# Save clustered dataset
df_main.to_csv('recotrip_places_with_clusters.csv', index=False)
print("✓ Saved: recotrip_places_with_clusters.csv")

print("\n✅ All data files and models created successfully!")
print(f"\nTotal places: {len(df_main)}")
print(f"Total states: {df_main['State'].nunique()}")
print(f"Clusters created: {df_main['Cluster'].nunique()}")
