import os 
from os import walk

def print_listImgs(imgList, folder):
    for idx, img in enumerate(imgList):
        print(f'from: {folder} \t idx: {idx} \t file name: {img}')

def rename_files(rootFolder, subFolder, destFolder, imgList: list):
    """
        rootFolder: path of folder where was saved all images (bw and bins)
        subFolder: name of subfolder inside `rootFolder`
        destFolder: path of folder to save the new images, renamed
        imgList: an array with the name of each image
    """
    for idx, img in enumerate(imgList):
        newName = ''
        if idx < 4:
            newName = img[0:11] + '.png'
        else:
            newName = img[0:10] + '.png'

        oldNameFile = os.path.join(rootFolder, subFolder, img)
        newNameFile = os.path.join(destFolder, newName)
        # os.rename(oldNameFile, newNameFile)

        # print(f'old name: {oldNameFile} / new: {newNameFile}')
        print(f'{newName}')

def main():

    rootFolder = '../contents/data_without_renamed_files'

    outFolders = [
        '../contents/data_with_renamed_files/image-bin-L1-renamed', 
        '../contents/zip_with_renamed_files/image-bin-L2-renamed', 
        '../contents/zip_with_renamed_files/image-bw-renamed'
    ]

    for (dirpath, dirnames, filenames) in walk(rootFolder):
        print(dirnames)
        for idx, actualDir in enumerate(dirnames):
            if (idx == 2):
                fullActualDir = os.path.join(rootFolder, actualDir)
                print(fullActualDir)
                for (acturalRoot, actualDirs, actualFiles) in walk(fullActualDir):
                    print(actualDir)
                    # print(actualFiles)
                    # print_listImgs(actualFiles, fullActualDir) 
                    rename_files(rootFolder, actualDir, outFolders[idx], actualFiles)
        break

if __name__ == "__main__":
    main()
