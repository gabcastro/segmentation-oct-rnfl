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