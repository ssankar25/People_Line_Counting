import os
import csv

# Directory where the image files are located
image_dir = 'TVHeads/segments/'
# Location to save the CSV file
csv_file = 'TVHeads/segments/labels.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the CSV file header
    writer.writerow(['filename', 'label'])

    # Traverse the image directory, read file names and labels
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Check file extension
            # Assuming the filename format is "label_XXX.ext"
            label = filename.split('_')[0]  # Extract the label
            writer.writerow([filename, label])

print('CSV file has been created')
