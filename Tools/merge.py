import pandas as pd
import glob
import os

# 1. Setup paths
# Update this to where your 6 separate files are located
source_folder = r"C:\Users\Abhishek\Desktop\New folder" 
# This matches the path in your config.py
output_file = r"C:\Users\Abhishek\Desktop\New folder\merged_data.csv"

# 2. Get all CSV files in the folder
all_files = glob.glob(os.path.join(source_folder, "*.csv"))

# 3. Read and combine them
# header=None is crucial because you said they don't have headers
df_list = [pd.read_csv(f, header=None) for f in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# 4. Save the merged file
# index=False and header=False keeps it clean for your dataset.py loader
combined_df.to_csv(output_file, index=False, header=False)

print(f"Successfully merged {len(all_files)} files into {output_file}")
print(f"Total rows: {len(combined_df)}")