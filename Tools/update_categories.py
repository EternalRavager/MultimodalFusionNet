import csv
import os

# File paths
merged_data_path = 'merged_data.csv'
merged_data2_path = 'merged_data2.csv'
output_path = 'merged_data_updated.csv'

print("Step 1: Loading category mapping from merged_data2.csv...")
# Create mapping from Category ID to Category Name (top-level only)
category_mapping = {}

with open(merged_data2_path, 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            category_id = row[0].strip()
            full_path = row[1].strip()
            
            # Extract top-level category name (before first slash)
            if '/' in full_path:
                category_name = full_path.split('/')[0]
            else:
                category_name = full_path
            
            category_mapping[category_id] = category_name

print(f"Loaded {len(category_mapping)} category mappings")

# Show sample mappings
print("\nSample category mappings:")
for i, (cat_id, cat_name) in enumerate(list(category_mapping.items())[:10]):
    print(f"  {cat_id} -> {cat_name}")

print("\nStep 2: Processing merged_data.csv...")
# Process merged_data.csv
processed_count = 0
unmapped_count = 0
unmapped_ids = set()

with open(merged_data_path, 'r', encoding='utf-8', errors='ignore') as infile, \
     open(output_path, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row_num, row in enumerate(reader, 1):
        if len(row) >= 4:
            url = row[0]
            title = row[1]
            description = row[2]
            category_id_raw = row[3].strip()
            
            # Normalize category ID (remove .0 if present)
            if category_id_raw.endswith('.0'):
                category_id = category_id_raw[:-2]
            else:
                category_id = category_id_raw
            
            # Map Category ID to Category Name
            if category_id in category_mapping:
                category_name = category_mapping[category_id]
            else:
                category_name = "Unknown"
                unmapped_count += 1
                unmapped_ids.add(category_id)
            
            # Write updated row
            writer.writerow([url, title, description, category_name])
            processed_count += 1
            
            if row_num % 100000 == 0:
                print(f"  Processed {row_num} rows...")
        else:
            print(f"Warning: Row {row_num} has insufficient columns, skipping")

print(f"\nStep 3: Complete!")
print(f"  Total rows processed: {processed_count}")
print(f"  Rows with unmapped categories: {unmapped_count}")

if unmapped_ids:
    print(f"\nUnmapped Category IDs (first 10): {list(unmapped_ids)[:10]}")

# Show category distribution
from collections import Counter
print("\nStep 4: Analyzing category distribution...")
category_counts = Counter()

with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 4:
            category_counts[row[3]] += 1

print("\nCategory distribution:")
for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {category}: {count:,} entries")

print(f"\n✓ Updated file saved as: {output_path}")
print("\nTo replace the original file, you can rename:")
print(f"  {output_path} -> {merged_data_path}")
