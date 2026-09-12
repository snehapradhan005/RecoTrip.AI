import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RecoTrip.ai - AI Travel Recommendations",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_models():
    df = pd.read_csv('recotrip_places_with_clusters.csv')
    kmeans = joblib.load('recotrip_kmeans_model.pkl')
    encoders = joblib.load('recotrip_encoders.pkl')
    season_encoder = joblib.load('recotrip_season_encoder.pkl')
    age_encoder = joblib.load('recotrip_age_encoder.pkl')
    scaler = joblib.load('recotrip_scaler.pkl')
    return df, kmeans, encoders, season_encoder, age_encoder, scaler

df, kmeans, encoders, season_encoder, age_encoder, scaler = load_models()

# Map duration buckets to an approximate hour value, matching how the
# training script derived Recommended_Duration_Days from actual hours.
DURATION_HOURS_MAP = {'0.5 Day': 3, '1 Day': 6, '2 Days': 12, '3+ Days': 20, 'Any': 8}


def build_user_vector(df, encoders, season_encoder, age_encoder, scaler,
                       budget, season, age_group, duration):
    """
    Turn the sidebar inputs into a feature vector in the exact same
    space script_1.py used to train the KMeans model:
    [Type_enc, Significance_enc, State_enc, Zone_enc, Budget_enc,
     season_onehot..., age_onehot..., rating, duration_hrs]

    The user doesn't supply Type/Significance/State/Zone, so we fill
    those with the dataset mode as a neutral placeholder.
    """
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

    # Use known seasons only; fall back to 'All Seasons' for any season
    # that wasn't in the training data (e.g., 'Autumn', 'Monsoon').
    known_seasons = set(season_encoder.classes_)
    season_query = [season] if (season != 'Any' and season in known_seasons) else ['All Seasons']
    known_ages = set(age_encoder.classes_)
    age_query = [age_group] if (age_group != 'Any' and age_group in known_ages) else ['All Age Groups']
    season_vec = season_encoder.transform([season_query])[0]
    age_vec = age_encoder.transform([age_query])[0]

    desired_rating = 4.5  # bias the query toward well-reviewed places
    duration_hrs = DURATION_HOURS_MAP.get(duration, 8)

    x_query = np.hstack([
        [type_enc, sig_enc, state_enc, zone_enc, budget_enc],
        season_vec,
        age_vec,
        [desired_rating, duration_hrs]
    ]).reshape(1, -1)

    return scaler.transform(x_query)

# Header
st.markdown('<h1 class="main-header">✈️ RecoTrip.ai</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Travel Recommendations Powered by Machine Learning</p>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("🎯 Your Travel Preferences")

    place_input = st.selectbox(
        "📍 Where do you want to go?",
        options=['Any'] + sorted(df['City'].unique().tolist()),
        help="Select your destination city"
    )

    duration = st.selectbox(
        "⏱️ How long will you stay?",
        options=['Any', '0.5 Day', '1 Day', '2 Days', '3+ Days'],
        help="Select your trip duration"
    )

    season = st.selectbox(
        "🌤️ Which season are you traveling?",
        options=['Any', 'Winter', 'Spring', 'Summer'],
        help="Select the season of your visit"
    )

    num_people = st.number_input(
        "👥 Number of travelers",
        min_value=1,
        max_value=20,
        value=2,
        help="How many people are traveling?"
    )

    budget = st.selectbox(
        "💰 Budget Category",
        options=['Any', 'Budget', 'Mid-Range', 'Premium'],
        help="Select your budget range"
    )

    age_group = st.selectbox(
        "👶 Age Group",
        options=['Any', 'Kids', 'Young Adults', 'Adults', 'Families', 'All Age Groups'],
        help="Select the age group of travelers"
    )

    st.markdown("---")
    recommend_button = st.button("🔍 Get Recommendations", use_container_width=True)

# Main content
if recommend_button:
    with st.spinner('🤖 AI is finding the best places for you...'):
        # Build a feature vector from the user's preferences and ask the
        # trained KMeans model which cluster of places best matches it.
        query_vec = build_user_vector(
            df, encoders, season_encoder, age_encoder, scaler,
            budget, season, age_group, duration
        )
        predicted_cluster = kmeans.predict(query_vec)[0]

        # Distance from the query to every cluster centroid, used to rank
        # places even when they aren't in the single nearest cluster.
        centroid_distances = kmeans.transform(query_vec)[0]

        filtered_df = df.copy()

        # ── Hard constraints ────────────────────────────────────────────────
        # Apply explicit user-selected filters BEFORE ML ranking so that the
        # ML only ranks within the valid candidate pool, not the whole dataset.

        # City filter
        if place_input != 'Any':
            filtered_df = filtered_df[filtered_df['City'] == place_input]

        # Season filter
        if season != 'Any':
            filtered_df = filtered_df[
                filtered_df['Ideal_Season'].str.contains(season, case=False, na=False) |
                filtered_df['Ideal_Season'].str.contains('All Seasons', case=False, na=False)
            ]

        # Budget filter
        if budget != 'Any':
            filtered_df = filtered_df[filtered_df['Budget_Category'] == budget]

        # Age group filter
        if age_group != 'Any':
            filtered_df = filtered_df[
                filtered_df['Age_Group_Suitability'].str.contains(age_group, case=False, na=False) |
                filtered_df['Age_Group_Suitability'].str.contains('All Age Groups', case=False, na=False)
            ]

        # ── ML ranking ──────────────────────────────────────────────────────
        # Now rank the filtered candidates using cluster distance from the ML
        # model — closer cluster centroid = more similar to user's preference.
        if len(filtered_df) > 0:
            filtered_df = filtered_df.copy()
            filtered_df['ml_distance'] = filtered_df['Cluster'].apply(
                lambda c: centroid_distances[c]
            )
            # Prefer places in the predicted cluster, then fall back to the
            # next-closest clusters instead of an all-or-nothing match.
            filtered_df = filtered_df.sort_values(
                ['ml_distance', 'Google review rating'],
                ascending=[True, False]
            )

        if len(filtered_df) == 0:
            st.error("😔 No places match your criteria. Try adjusting your filters!")
        else:
            st.success(f"🎉 Found {len(filtered_df)} amazing places for you!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📍 Places Found", len(filtered_df))
            with col2:
                avg_rating = filtered_df['Google review rating'].mean()
                st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")
            with col3:
                avg_fee = filtered_df['Entrance Fee in INR'].mean()
                st.metric("💵 Avg Entry Fee", f"₹{avg_fee:.0f}")
            with col4:
                states_count = filtered_df['State'].nunique()
                st.metric("🗺️ States Covered", states_count)

            st.markdown("---")
            st.subheader("🎯 Top Recommendations Based on Your Preferences")

            results_df = filtered_df.head(10)[[
                'Name', 'City', 'State', 'Type', 'Google review rating',
                'Entrance Fee in INR', 'Budget_Category', 'Ideal_Season',
                'Age_Group_Suitability', 'Best Time to visit', 'time needed to visit in hrs'
            ]].copy()

            results_df = results_df.rename(columns={
                'Name': '🏛️ Place Name',
                'City': '📍 City',
                'State': '🗺️ State',
                'Type': '🏷️ Type',
                'Google review rating': '⭐ Rating',
                'Entrance Fee in INR': '💰 Entry Fee (₹)',
                'Budget_Category': '💵 Budget',
                'Ideal_Season': '🌤️ Best Season',
                'Age_Group_Suitability': '👥 Suitable For',
                'Best Time to visit': '⏰ Best Time',
                'time needed to visit in hrs': '⏱️ Duration (hrs)'
            })

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Recommendations as CSV",
                data=csv,
                file_name=f"recotrip_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.subheader("🌟 Detailed Information - Top 3 Picks")

            for idx, row in filtered_df.head(3).iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>🏛️ {row['Name']}</h3>
                        <p><strong>📍 Location:</strong> {row['City']}, {row['State']} ({row['Zone']})</p>
                        <p><strong>🏷️ Type:</strong> {row['Type']} | <strong>✨ Significance:</strong> {row['Significance']}</p>
                        <p><strong>⭐ Rating:</strong> {row['Google review rating']}/5.0 ({row['Number of google review in lakhs']} Lakh reviews)</p>
                        <p><strong>💰 Entry Fee:</strong> ₹{row['Entrance Fee in INR']} ({row['Budget_Category']})</p>
                        <p><strong>⏱️ Time Needed:</strong> {row['time needed to visit in hrs']} hours | <strong>⏰ Best Time:</strong> {row['Best Time to visit']}</p>
                        <p><strong>🌤️ Ideal Season:</strong> {row['Ideal_Season']}</p>
                        <p><strong>👥 Suitable For:</strong> {row['Age_Group_Suitability']}</p>
                        <p><strong>📸 DSLR Allowed:</strong> {row['DSLR Allowed']} | <strong>✈️ Airport Nearby:</strong> {row['Airport with 50km Radius']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")

else:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🤖 AI-Powered
        Our machine learning algorithm analyzes your preferences to suggest the perfect destinations
        """)

    with col2:
        st.markdown("""
        ### 🎯 Personalized
        Get recommendations tailored to your budget, time, season, and travel group
        """)

    with col3:
        st.markdown("""
        ### 📊 Data-Driven
        Based on ratings and reviews from millions of travelers across India
        """)

    st.markdown("---")
    st.info("👈 Fill in your travel preferences in the sidebar and click '🔍 Get Recommendations' to start!")

    st.subheader("📈 Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏛️ Total Places", len(df))
    with col2:
        st.metric("🗺️ States Covered", df['State'].nunique())
    with col3:
        st.metric("🏙️ Cities", df['City'].nunique())
    with col4:
        st.metric("🏷️ Place Types", df['Type'].nunique())

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>✈️ RecoTrip.ai - Your Intelligent Travel Companion | Powered by Machine Learning 🤖</p>
    <p>Discover India's Hidden Gems with AI-Powered Recommendations</p>
</div>
""", unsafe_allow_html=True)
