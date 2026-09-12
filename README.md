# RecoTrip.ai - Travel Recommendation System

A machine learning-based travel recommendation tool that suggests tourist destinations across India based on user preferences. Built with Streamlit for the frontend and K-Means clustering for the recommendation logic.

---

## About

This project started as an attempt to solve a simple but common problem: **planning a trip is overwhelming**. There are too many options, too many factors to consider (budget, season, duration, who you're traveling with), and too little guidance on where to start.

Instead of building another travel search engine, I focused on **filtering and ranking** places based on practical constraints. The dataset covers 325+ tourist spots across 33 Indian states, each tagged with attributes like:

- Zone and city
- Best season to visit
- Suitable age groups
- Budget category (Budget / Mid-Range / Premium)
- Google review ratings
- Entrance fees

The recommendation engine uses **K-Means clustering (15 clusters)** to group similar destinations, combined with a filtering pipeline that narrows down options based on what you actually care about — where you want to go, how long you're staying, the season, group size, budget, and age group.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| ML Libraries | scikit-learn, pandas, NumPy, joblib |
| Frontend | Streamlit |
| Algorithm | K-Means Clustering (k=15) |

---

## How It Works

1. **Data Preparation** — The dataset (`recotrip_dataset.csv`) was cleaned and enriched with additional features including `Budget_Category`, `Season_Suitability`, and `Age_Group_Suitability`. K-Means clustering was applied to group 325+ places into 15 meaningful clusters, and the results were saved to `recotrip_places_with_clusters.csv`.

2. **Model Pickling** — Trained models and encoders (K-Means model, LabelEncoders for season/age, StandardScaler, and one-hot encoders) are serialized as `.pkl` files for fast runtime loading.

3. **Inference Pipeline** — At runtime, the app loads all pre-trained models using `@st.cache_resource`, applies user-selected filters, and returns a ranked list of destinations sorted by Google review rating.

4. **Output** — Results are displayed in a styled table with the option to download as CSV. The average entrance fee for the recommended cohort is also calculated.

---

## Features

- **6 input parameters**: Destination city, trip duration (0.5–3+ days), season, group size (1–20), budget tier, and age group
- **325+ places** across 33 Indian states, organized into 15 K-Means clusters
- **Filtering-based ranking**: Places are filtered by user constraints and ranked by Google review rating
- **Budget transparency**: Average entrance fee displayed for the filtered results
- **CSV export**: Download recommendation results for offline reference
- **Lightweight**: Runs locally with no external API dependencies

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
pip install streamlit pandas scikit-learn joblib numpy
```

### Run the App

```bash
cd recotrip
streamlit run recotrip_app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## Project Structure

```
recotrip/
├── recotrip_app.py              # Streamlit frontend + inference logic
├── recotrip_dataset.csv         # Raw tourist destination data
├── recotrip_places_with_clusters.csv  # Dataset with K-Means cluster assignments
├── recotrip_kmeans_model.pkl    # Trained K-Means model
├── recotrip_encoders.pkl        # Encoders for categorical features
├── recotrip_season_encoder.pkl  # Season-specific encoder
├── recotrip_age_encoder.pkl     # Age-group encoder
├── recotrip_scaler.pkl          # Feature scaler
├── script.py                    # Main data preprocessing pipeline
├── script_1.py - script_4.py    # Supporting scripts for data prep
└── README.md
```

---

## Limitations

This is a **filtering and clustering-based system**, not a neural network or transformer model. The recommendations are only as good as the data and the rules defined. Some known limitations:

- No real-time pricing or availability data
- No personalization across sessions (no user history or learning)
- Budget is categorical (Budget / Mid-Range / Premium) rather than a continuous numeric value
- Does not account for travel time between locations or route optimization
- Dataset is static and manually curated

These are intentional trade-offs for this project scope. Building a full-fledged travel planner would require real-time APIs, a more sophisticated recommendation model, and a larger dataset.

---

## What I Learned

- How to structure a complete ML pipeline from raw data to deployment
- Working with K-Means clustering for grouping similar entities
- Building an interactive UI with Streamlit
- Handling categorical data with encoders and scalers
- Balancing model complexity with practical usability

---

## License

MIT License — feel free to use, modify, and experiment.
