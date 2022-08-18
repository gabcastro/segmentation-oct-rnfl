import os 
from os import walk

def print_listImgs(imgList, folder):
    for idx, img in enumerate(imgList):
        print(f'from: {folder} \t idx: {idx} \t file name: {img}')

def rename_files(rootFolder, subFolder, destFolder, imgList):
    """
        rootFolder: path of folder where was saved all images (bw and bins)
        subFolder: name of subfolder inside `rootFolder`
        destFolder: path of folder to save the new images, renamed
        imgList: an array with the name of each image
    """
    for idx, img in enumerate(imgList):
        new_name = ''
        if idx < 4:
            new_name = img[0:11] + '.png'
        else:
            new_name = img[0:10] + '.png'
        os.rename(rootFolder + subFolder + '\\' + img, rootFolder + subFolder + '\\' + destFolder + new_name)
        print(f'{idx} - file {img} renamed to {new_name} ...')

def main():

    root_folder = 'F:\\documents\\unisinos\\master\\!research-images\\IMAGES_DEID_WITH_ANOTATIONS - PS editions\\Exports-ROIs-square-512\\'

    renamed_folders_out = ['image-bin-L1-renamed\\', 'image-bin-L2-renamed\\', 'image-bw-renamed\\']

    for (dirpath, dirnames, filenames) in walk(root_folder):
        for idx, subdir in enumerate(dirnames):
            for (dirpath, dirnames, filenames) in walk(root_folder + subdir):
                # print_listImgs(filenames, subdir) 
                rename_files(root_folder, subdir, renamed_folders_out[idx], filenames)

if __name__ == "__main__":
    main()
