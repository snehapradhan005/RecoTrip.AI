
# Now let's create the documentation files and ZIP everything together
import zipfile
import os

# Create the README
readme_content = '''# ✈️ RecoTrip.ai - AI Travel Recommendation System

## Quick Start Guide

### Step 1: Install Python Packages
```bash
pip install streamlit pandas scikit-learn joblib numpy
```

### Step 2: Extract ZIP File
Extract all files to a folder named "recotrip" on your Desktop

### Step 3: Run the App
```bash
cd Desktop/recotrip
streamlit run recotrip_app.py
```

## Required Files (All 8 are included in this ZIP)
1. recotrip_app.py
2. recotrip_dataset.csv
3. recotrip_places_with_clusters.csv
4. recotrip_kmeans_model.pkl
5. recotrip_encoders.pkl
6. recotrip_season_encoder.pkl
7. recotrip_age_encoder.pkl
8. recotrip_scaler.pkl

## Features
- 6 Input Parameters (Place, Duration, Season, People, Budget, Age)
- 325+ Places across 33 Indian States
- ML-Powered Recommendations (15 Clusters)
- Table Output Format
- Download Results as CSV

## Troubleshooting
- Python not found: Install from python.org
- streamlit not found: Run pip install streamlit again
- File errors: Make sure all 8 files are in same folder

Good luck with your project! 🚀
'''

with open('README.txt', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("✓ Created README.txt")

# Create installation guide
install_guide = '''
╔═══════════════════════════════════════════════════════════════╗
║         RECOTRIP.AI - INSTALLATION INSTRUCTIONS              ║
╚═══════════════════════════════════════════════════════════════╝

STEP 1: Extract the ZIP File
────────────────────────────
1. Right-click on the ZIP file you downloaded
2. Select "Extract All" or "Extract Here"
3. Choose Desktop as location
4. Rename extracted folder to: recotrip


STEP 2: Install Python (if not installed)
──────────────────────────────────────────
1. Go to: https://www.python.org/downloads/
2. Download latest Python (3.7 or higher)
3. Run installer
4. ✓ CHECK "Add Python to PATH" (IMPORTANT!)
5. Click Install


STEP 3: Install Required Packages
───────────────────────────────────
1. Press Windows Key + R
2. Type: cmd
3. Press Enter
4. Copy-paste this command:

   pip install streamlit pandas scikit-learn joblib numpy

5. Press Enter and wait 2-3 minutes


STEP 4: Navigate to Project Folder
────────────────────────────────────
In the same Command Prompt window, type:

   cd Desktop/recotrip

Press Enter


STEP 5: Run the Application
─────────────────────────────
Type this command:

   streamlit run recotrip_app.py

Press Enter - browser will open in 10 seconds!


STEP 6: Use the App
────────────────────
1. Fill inputs in left sidebar
2. Click "Get Recommendations"
3. See results in table!


═══════════════════════════════════════════════════════════════

That's it! If you need help, email or message me.

🎉 Enjoy RecoTrip.ai!
'''

with open('INSTALLATION.txt', 'w', encoding='utf-8') as f:
    f.write(install_guide)

print("✓ Created INSTALLATION.txt")

# Now create the ZIP file with all essentials
zip_filename = 'RecoTrip_AI_Complete_Package.zip'

files_to_include = [
    'recotrip_app.py',
    'recotrip_dataset.csv',
    'recotrip_places_with_clusters.csv',
    'recotrip_kmeans_model.pkl',
    'recotrip_encoders.pkl',
    'recotrip_season_encoder.pkl',
    'recotrip_age_encoder.pkl',
    'recotrip_scaler.pkl',
    'README.txt',
    'INSTALLATION.txt'
]

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for filename in files_to_include:
        if os.path.exists(filename):
            zipf.write(filename, filename)
            file_size = os.path.getsize(filename)
            print(f"✓ Added: {filename} ({file_size/1024:.1f} KB)")

print(f"\n{'='*65}")
print(f"🎉 SUCCESS! Created Complete Package!")
print(f"{'='*65}")
print(f"📦 File: {zip_filename}")
print(f"📊 Size: {os.path.getsize(zip_filename)/(1024*1024):.2f} MB")
print(f"📁 Contains: {len(files_to_include)} files")
print(f"{'='*65}")
print("\n✅ This single ZIP file contains EVERYTHING you need!")
print("✅ Just download it, extract, and follow INSTALLATION.txt")
