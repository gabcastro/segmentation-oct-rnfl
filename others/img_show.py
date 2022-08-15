import cv2
from skimage import io

def main():
    folder = 'E:\\documents\\unisinos\\master\\research\\IMAGES_DEID_ANOTATIONS\\IMAGES_DEID_P1\\IMAGES_DEID_P1\\001\\'
    im = '001_OE_ONH.jpg'

    image = io.imread(folder + im)
    # image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    # image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    print(image.shape)

    cv2.imshow('', image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()