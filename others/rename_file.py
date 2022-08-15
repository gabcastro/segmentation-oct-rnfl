import os 
from os import walk

def main():

    folder = 'E:\\documents\\unisinos\\master\\research\\IMAGES_DEID_ANOTATIONS\\contexto_1_512__adjusts\\'
    new_folder_mask = 'anotations\\'
    new_folder_orig = 'renamed\\'

    allImgs = []
    maskImgs = []
    normalImgs = []

    for (dirpath, dirnames, filenames) in walk(folder):
            allImgs.extend(filenames)
            break

    normalImgs = allImgs    

    # for idx, img in enumerate(allImgs):
    #     if img.find("Camada") != -1:
    #         maskImgs.append(img)
    #     elif img.find("Plano-de-Fundo") != -1:
    #         normalImgs.append(img)

    print("\n*********** PRINT INFOS ABOUT FILES ***************\n")
    # print_listImgs(maskImgs)
    # print("\n**********************************\n")
    print_listImgs(normalImgs)

    print("\n************ RENAMING FILES ****************\n")
    # rename_files(new_folder_mask, maskImgs, folder)
    # print("\n**********************************\n")
    rename_files(new_folder_orig, normalImgs, folder)
    

def print_listImgs(imgList):
    for idx, img in enumerate(imgList):
        print(f'idx: {idx}, file name: {img}')

def rename_files(destFolder, imgList, srcFolder):
    for idx, img in enumerate(imgList):
        new_name = ''
        if idx < 4:
            new_name = img[0:11] + '.png'
        else:
            new_name = img[0:10] + '.png'
        os.rename(srcFolder + img, srcFolder + destFolder + new_name)
        print(f'{idx} - file {img} renamed to {new_name} ...')


if __name__ == "__main__":
    main()
