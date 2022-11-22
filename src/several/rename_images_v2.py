import os
import shutil
from glob import glob

ROOT_DIR = "../../data/all_data_640"
DATA_DIR = "../../data/all_data_640/all_images"

IDS = ["background", "layer_1", "layer_2"]


def load_images(root_dir, folder_imgs):
    return sorted(s[-1] for s in [f.split('/') for f in glob(os.path.join(root_dir, folder_imgs))])


def find_images(data, substring):
    new_list = []
    for d in data:
        if substring in d:
            new_list.append(d)

    return new_list


def rename_name(file_name, substring):
    # 6 because the name contains a code _0001_ or _0000_ after _ONH
    # 4 because the name contains an extension .png
    l = (10 + len(substring)) * -1
    return file_name[:l] + ".png"


def op_move(data, image_type):
    print(f"Copy {image_type} images ... ")
    for i in data:
        oldname = os.path.join(DATA_DIR, i)
        newname = rename_name(i, image_type)
        newname = os.path.join(ROOT_DIR, image_type, newname)

        shutil.copy2(oldname, newname)
    print("... done!")

def rename():
    data = load_images(DATA_DIR, "*.png")

    background_data = find_images(data, "background")
    layer_1_data = find_images(data, "layer_1")
    layer_2_data = find_images(data, "layer_2")
    
    op_move(background_data, IDS[0])
    op_move(layer_1_data, IDS[1])
    op_move(layer_2_data, IDS[2])
    
if __name__ == "__main__":
    rename()