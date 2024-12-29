import os
import csv


def enumerate_files_in_folder(folder_path, output_csv):
    # Get a list of all files in the folder
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Write the filenames to a CSV file
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename"])  # Write the header
        for filename in files:
            writer.writerow([filename])


# Example usage
folder_path = "./A Very Marmot Christmas/outputs"  # Replace with your folder path
output_csv = "file_list.csv"  # Replace with your desired output CSV file name
enumerate_files_in_folder(folder_path, output_csv)
