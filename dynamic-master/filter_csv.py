import os
import csv

video_dir = "echonet/data"
input_csv = os.path.join(video_dir, "FileList_.csv")
output_csv = os.path.join(video_dir, "FileList.csv")

# Read existing .avi filenames (with or without extension)
avi_files = {os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith(".avi")}

print(f"Found {len(avi_files)} .avi files.")

with open(input_csv, newline='') as infile, open(output_csv, "w", newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    count = 0
    for row in reader:
        filename = row["FileName"].strip()
        if filename in avi_files:
            writer.writerow(row)
            count += 1
        else:
            print(f"Skipping missing video: {filename}")

print(f"✅ Done. Kept {count} valid entries in FileList.csv.")
