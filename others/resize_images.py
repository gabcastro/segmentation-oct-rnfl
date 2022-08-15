from PIL import Image
from os import walk

def resize_imgs(mypath, mypathDest, f):

    for fileName in f:
        img = Image.open(mypath + fileName)
        img.thumbnail((512, 512))
        img.save(mypathDest + fileName)


def walk_folder(mypath):
    f = []
    
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break

    return f

def main():

    mainFolderOrig = 'rnfl_dataset_context_3\\'
    mainFolderDest = 'rnfl_dataset_context_3_resize\\'

    layer = 'layer_1\\'
    folderTrain = 'train\\'
    folderTest = 'test\\'
    
    folderOrig = 'original_images\\'
    folderGT = 'ground_truth\\'
    

    mypathOrig = [
        mainFolderOrig + layer + folderTrain + folderOrig,
        mainFolderOrig + layer + folderTrain + folderGT,
        mainFolderOrig + layer + folderTest + folderOrig,
        mainFolderOrig + layer + folderTest + folderGT
    ]

    mypathDest = [
        mainFolderDest + layer + folderTrain + folderOrig,
        mainFolderDest + layer + folderTrain + folderGT,
        mainFolderDest + layer + folderTest + folderOrig,
        mainFolderDest + layer + folderTest + folderGT
    ]
    
    for index, i in enumerate(mypathOrig):
        mypath = i
        mypathDestt = mypathDest[index]

        print(mypath)
        print(mypathDestt)

        f = walk_folder(mypath)
        resize_imgs(mypath, mypathDestt, f)

        print('done...')

    print('complete')


if __name__ == "__main__":
    main()