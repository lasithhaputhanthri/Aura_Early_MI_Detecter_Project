# Save this as quick_check.py and run it
import pandas as pd
import os

# Check CSV content
try:
    filelist = pd.read_csv('echonet/data/FileList.csv')
    print(f"FileList.csv: {len(filelist)} rows")
    print("Columns:", list(filelist.columns))
    print("First few rows:")
    print(filelist.head())
    
    tracings = pd.read_csv('echonet/data/VolumeTracings.csv')
    print(f"\nVolumeTracings.csv: {len(tracings)} rows")
    print("Columns:", list(tracings.columns))
    
    # Check video files
    videos = os.listdir('echonet/data/Videos')
    avi_files = [f for f in videos if f.endswith('.avi')]
    print(f"\nVideo files: {len(avi_files)} .avi files")
    
    if len(avi_files) > 0:
        print("First few video files:", avi_files[:5])
    
except Exception as e:
    print(f"Error: {e}")