import os
import csv
from glob import glob

DATA_DIR = [
    '../../data/v2/L1/train',
    '../../data/v2/L1/test',
    '../../data/v2/L1/validation'
]

def list_images():
    """Read a specific folder and creat a csv file with file names
    """
    images = load_data(DATA_DIR, "images/*.png")

    with open('../../data/map_images/output.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter = '\n')
        writer.writerow(images)

    print('data created...')
    
def load_data(data_dir, folder_img):
    data = []

    for dir in data_dir:
        data += sorted(s[-1] for s in [f.split('/') for f in glob(os.path.join(dir, folder_img))])

    return data

if __name__ == "__main__":
    list_images()