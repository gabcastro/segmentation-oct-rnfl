import os
import csv
import shutil

SRC_DIR = '../../data/all_data_640'
DST_DIR = '../../data/v4'

MASK_FOLDER = ['layer_1', 'layer_2']

def create_split_dataset():
    """Search images from folder with all images and split in
    train, test and validation.

    Split is set by default in 70% to train, and 15% to validation and test.
    """
    data = []
    
    print('reading csv...')
    # read csv with: file name and an attribute 'marked' (Y or N)
    with open("../../data/map_images/imgs_marked.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            x = row[0].split(';')
            if x[-1] == 'Y':
                data.append(x[0])
    print('csv readed')

    # split dataset as mentioned above
    x_total = len(data)
    x_train = int(x_total * 0.7)
    x_test = int((x_total - x_train) * 0.5)
    x_val = x_total - x_train - x_test

    print(f'**** Total images: {x_total}')
    print(f'**** Train: {x_train}')
    print(f'**** Test: {x_test}')
    print(f'**** Validation: {x_val}')

    img_train = data[:x_train]
    img_test = data[x_train:(x_train + x_test)]
    img_val = data[x_total - x_val:]

    copy(img_train, 1)
    copy(img_test, 2)
    copy(img_val, 3)

def copy(data, to):
    folder = ''

    if to == 1:
        folder = 'train'
    elif to == 2:
        folder = 'test'
    else:
        folder = 'validation'

    print(f'====> copy to folder {folder} will start')

    for img_name in data:
        # get grayscale image
        img_from = os.path.join(SRC_DIR, 'background', img_name)

        # to each layer, get mask image
        for idx, m in enumerate(MASK_FOLDER):
            mask_from = os.path.join(SRC_DIR, m, img_name)

            # set new directory to grayscale and binary images
            l = (f'L{idx + 1}')
            img_to = os.path.join(DST_DIR, l, folder, 'images')
            mask_to = os.path.join(DST_DIR, l, folder, 'masks')

            try:
                shutil.copy2(img_from, img_to)
                shutil.copy2(mask_from, mask_to)
                
                print(f'file {img_name} copied')
            except:
                print(f'wasnt possible copy file: {img_name}')

    print('copy finished')

if __name__ == "__main__":
    create_split_dataset()