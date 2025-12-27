import csv
import glob
import os

def convert_tsv_to_csv():
    # Find all .tsv files in the current directory
    tsv_files = glob.glob("*.tsv")
    
    if not tsv_files:
        print("No .tsv files found in this directory.")
        return

    print(f"Found {len(tsv_files)} TSV files. Starting conversion...")

    for tsv_file in tsv_files:
        csv_file = tsv_file.replace('.tsv', '.csv')
        print(f"Converting '{tsv_file}' to '{csv_file}'...")
        
        try:
            with open(tsv_file, mode='r', encoding='utf-8', newline='') as infile:
                with open(csv_file, mode='w', encoding='utf-8', newline='') as outfile:
                    # Read as Tab-Separated
                    reader = csv.reader(infile, delimiter='\t')
                    # Write as Comma-Separated
                    writer = csv.writer(outfile)
                    
                    for row in reader:
                        writer.writerow(row)
            
            print(f"Successfully created '{csv_file}'")
            
        except Exception as e:
            print(f"Error converting {tsv_file}: {e}")

    print("All done!")

if __name__ == "__main__":
    convert_tsv_to_csv()