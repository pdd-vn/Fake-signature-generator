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
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def psuedo_binary_img(img, threshold=200):
    '''
    create binary image in gray scale format.
    input: 2D image.
    output: 2D image which only take 0 or 255 value.
    '''
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j]<threshold:
                img[i][j]=0
            else:
                img[i][j]=255
    return img


def hrz_border(img):
    '''
    get horizontal border.
    input: 2D image.
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
    input: 2D image.
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


def augment(img, color, blur_rate=8, erosion_kernel_size=2, num_eigenvalues=30):
    '''
    but nhoe muc
    '''
    # read image
    img = np.array(img)
    img = psuedo_binary_img(img)

    # create blur erosion
    kernel_size = erosion_kernel_size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(img, kernel)
    blur = iaa.blur.AverageBlur(blur_rate)
    blur_erosion = blur(image=erosion)
    
    # get signature position by pixel
    sig_pos = np.where(blur_erosion != 255)

    # drop image information using svd
    img = svd(img, num_eigenvalues)

    # create eroded image
    erosion = cv2.erode(img, kernel)

    # blur erosion
    blur = iaa.blur.AverageBlur(blur_rate)
    blur_erosion = blur(image=erosion)
    
    # overide blur erosion 
    override_pos = np.where(img <= blur_erosion)
    blur_erosion[override_pos] = img[override_pos]
    
    # create clean signature for transparent
    holder = np.zeros(blur_erosion.shape)
    holder[:] = 255
    holder[sig_pos] = blur_erosion[sig_pos]
    
    rgb_img = augment_change_color(holder, sig_pos, color)
    import ipdb; ipdb.set_trace()
    return rgb_img

def augment_change_color(img, sig_pos, color, enhanced_value=0.8):
    h, w = img.shape
    rgb_img = np.zeros((h,w,3))
    rgb_img[:,:,0] = img/255
    rgb_img[:,:,1] = img/255
    rgb_img[:,:,2] = img/255
    if color == "blue":
        rgb_img[:,:,2][sig_pos] = enhanced_value
        rgb_img[rgb_img<0] = 0
    elif color == "red":
        rgb_img[:,:,0][sig_pos] = enhanced_value
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
    augment(img, 'blue')