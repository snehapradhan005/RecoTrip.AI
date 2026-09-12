
import zipfile
import os

# Create a list of all the files we need to zip
files_to_zip = [
    'recotrip_app.py',
    'recotrip_dataset.csv',
    'recotrip_places_with_clusters.csv',
    'recotrip_kmeans_model.pkl',
    'recotrip_encoders.pkl',
    'recotrip_season_encoder.pkl',
    'recotrip_age_encoder.pkl',
    'recotrip_scaler.pkl',
    'RECOTRIP_README.md',
    'INSTALLATION_GUIDE.txt',
    'PROJECT_SUMMARY.txt',
    'QUICK_START.txt'
]

# Create a ZIP file
zip_filename = 'RecoTrip_AI_Complete_Package.zip'

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file, file)
            print(f"✓ Added: {file}")
        else:
            print(f"✗ Missing: {file}")

print(f"\n🎉 SUCCESS! Created: {zip_filename}")
print(f"📦 File size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
print("\nThis ZIP contains all 8 essential files + documentation!")
