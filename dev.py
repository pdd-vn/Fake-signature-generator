import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from imgaug import augmenters as iaa
from tqdm import tqdm


def read_gray(path):
    '''
    read image in gray scale.
    input: image path.
    output: gray image in cv2 format.
    '''
    img = cv2.imread(path, 0)
    return img


def psuedo_binary_img(img, threshold=200):
    '''
    create binary image in gray scale format.
    input: gray image.
    output: gray image which only take 0 or 255 value.
    '''
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh


def hrz_border(img):
    '''
    get horizontal border.
    input: gray image.
    output: upper and lower border of input.
    '''
    h, w = img.shape
    for i in range(h):
        if np.mean(img[i])!=255:
            upper = i
            break
    for i in range(h-1,0,-1):
        if np.mean(img[i])!=255:
            lower = i
            break
    return upper, lower


def crop_border(img):
    '''
    crop image from its border.
    input: gray image.
    output: cropped image.
    '''
    upper, lower = hrz_border(img)
    h, w = img.shape
    result = img[upper:lower,]
    return result


def svd(img, k):
    '''
    singular value decomposition
    input: - img: your image in gray scale
            - k: number of kept eigenvalues
    output: truncated image.
    '''
    u, s, v = np.linalg.svd(img)
    return np.dot(u[:,:k],np.dot(np.diag(s[:k]),v[:k,:]))


def create_thick_blur_sig(img, blur_kernel_size, erosion_kernel_size):
    '''
    input: - img: signature gray image.
           - blur_kernel_size: cv2.blur ksize.
           - erosion_kernel_size: cv2.erode kernel size.
    output: thick blur signature image.
    '''
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    erosion = cv2.erode(img, kernel)
    thick_blur_sig = cv2.blur(erosion, (blur_kernel_size, blur_kernel_size))

    return thick_blur_sig


def augment_sig(img, blur_kernel_size=8, erosion_kernel_size=2, num_eigenvalues=30):
    '''
    create real signature with white background for transparent.
    input: - img: gray img.
           - blur_kernel_size: cv2.blur ksize.
           - erosion_kernel_size: cv2.erode kernel size.
    output: signature with shadow for more realistic. 
    '''
    # force gray image to contain only 0 and 255 value, aka psuedo binary image
    img = psuedo_binary_img(img)

    # create shadow and get whole signature position by pixel using shadow
    shadow = create_thick_blur_sig(img, blur_kernel_size, erosion_kernel_size)
    sig_pos = np.where(shadow != 255)

    # drop image information using svd
    truncated_img = svd(img, num_eigenvalues)

    # create shadow for truncated signature
    truncated_shadow = create_thick_blur_sig(truncated_img,
                                            blur_kernel_size,
                                            erosion_kernel_size)
    
    # overide shadow
    overwrite_pos = np.where(truncated_img <= truncated_shadow)
    truncated_shadow[overwrite_pos] = truncated_img[overwrite_pos]
    
    # create clean signature for transparent
    clean_sig = np.zeros(img.shape)
    clean_sig[:] = 255
    clean_sig[sig_pos] = truncated_shadow[sig_pos]
    
    return clean_sig



def augment_change_color(img, sig_pos, color, enhanced_value=0.8):
    '''
    change color of augmented signature.
    input: - img: signature to change color.
           - sig_pos: position of signature by pixel.
           - color: color to be changed to. Available colors are red, blue and black.
           - enhanced_value: value to be set to enhanced strength of a channel.
    output: RGB signature.
    '''
    # h, w = img.shape
    # rgb_img = np.zeros((h,w,3))
    # rgb_img[:,:,0] = img/255
    # rgb_img[:,:,1] = img/255
    # rgb_img[:,:,2] = img/255
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if color == "blue":
        rgb_img[:,:,2][sig_pos] = enhanced_value * 255
        rgb_img[rgb_img<0] = 0
    elif color == "red":
        rgb_img[:,:,0][sig_pos] = enhanced_value * 255
        rgb_img[rgb_img<0] = 0
    elif color == "black":
        rgb_img[rgb_img<0]=0
    else:
        raise TypeError("Only support blue, red or black color")

    return rgb_img


def show_rgb(img):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()


def show(img):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__=="__main__":
    img = read_gray('/home/pdd/Desktop/workspace/sig_gen/Fake-Data-Generator/input_signature/sig_0.png')
    import ipdb; ipdb.set_trace()