import cv2
import numpy as np

def flip(img, orientation):
    """Flip an image horizontally, vertically, or both horizontally and vertically
    
    Args:
        orientation: integer
            -1: horizontally and vertically
            0: vertically
            1: horizontally 
    """
    i_aug = cv2.flip(img, orientation)

    return i_aug

def rotation_zoom(img, scale, angle=0):
    """Applies a rotation or zoom operation in an image.
    To rotatation the parameter `angle` is more important than `scale`.
    For scale, `angle` is zero.
    """
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # rotate the image by an angle around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    img_rotated = cv2.warpAffine(img, M, (w, h))

    return img_rotated

def shift(img, tx, ty):
    """Applies the operation shift to an image. To perform, is use an image translation, 
    called affine transformation matrix.
    
    For the purposes of translation, all we care about are the `tx` and `ty` values:

        - Negative values for the tx value will shift the image to the left
        - Positive values for tx shifts the image to the right
        - Negative values for ty shifts the image up
        - Positive values for ty will shift the image down
    """
    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return shifted

def ilumi_normalization(img):
    """Based on stack and paper: 
        - https://stackoverflow.com/questions/62441106/illumination-normalization-using-python-opencv
        - https://arxiv.org/pdf/1907.09449.pdf

        TODO: still necessary read better the paper and see what I can change
    """
    # illumination normalize
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # separate channels
    y, cr, cb = cv2.split(ycrcb)

    # get background which paper says 
    # (gaussian blur using standard deviation 5 pixel for HxW size image)
    # account for size of input vs 300
    sigma = int(5 * 512 / 100)
    gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

    # subtract background from Y channel
    y = (y - gaussian + 100)

    # merge channels back
    ycrcb = cv2.merge([y, cr, cb])

    #convert to BGR
    output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return output
