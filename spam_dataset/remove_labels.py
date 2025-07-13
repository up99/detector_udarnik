# This script remove labels from coco128. Then these images will be added to source_dataset to expand diversity of dataset.

import os

directory = 'coco128/labels/train2017/'

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as file:
            pass  

print("All .txt files have been emptied.")