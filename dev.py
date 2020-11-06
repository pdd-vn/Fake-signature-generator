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
    
    rgb = augment_change_color(holder, sig_pos, color)

    return rgb

def augment_change_color(holder, sig_pos, color):
    if color == "blue":
        h, w = holder.shape
        enhanced = holder/255
        enhanced[sig_pos]=0.8
        rgb = np.zeros((h,w,3))
        rgb[:,:,0]=holder/255
        rgb[:,:,1]=holder/255
        rgb[:,:,2]=enhanced
        rgb[rgb<0]=0
    elif color == "red":
        h, w = holder.shape
        enhanced = holder/255
        enhanced[sig_pos]=0.8
        rgb = np.zeros((h,w,3))
        rgb[:,:,0]=enhanced
        rgb[:,:,1]=holder/255
        rgb[:,:,2]=holder/255
        rgb[rgb<0]=0
    elif color == "black":
        h, w = holder.shape
        rgb = np.zeros((h,w,3))
        rgb[:,:,0]=holder/255
        rgb[:,:,1]=holder/255
        rgb[:,:,2]=holder/255
        rgb[rgb<0]=0
    else:
        raise TypeError("Only support blue, red or black color")

    return rgb

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

def extract_sig(raw_sig_path):
    import glob
    from shutil import copyfile as cp

    for idx,folder in enumerate(glob.glob(raw_sig_path)):
        try:
            img_path = glob.glob(folder+"/*")[0]
            cp(img_path, "./input_signature/sig_{}.png".format(idx))
        except: 
            pass


def main(args):
    img_list = []
    data_path = args.input_signature
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if len(os.listdir(folder_path)) > 2:
            choosen = random.choices(os.listdir(folder_path), k=2)
            choosen[0] = os.path.join(folder_path,choosen[0])
            choosen[1] = os.path.join(folder_path,choosen[1])
            img_list.append(choosen)
        elif len(os.listdir(folder_path))==1:
            choosen = os.path.join(folder_path, os.listdir(folder_path)[0])
            img_list.append(choosen)
        else:
            pass
    
    # import ipdb; ipdb.set_trace()
    img_list = sum(img_list, [])

    for i in tqdm(range(len(img_list))):
        img = read_gray(img_list[i])
        img = augment(img, blur_rate=10, erosion_kernel_size=1)
        out_path = os.path.join(output_folder, "{}.png".format(i))
        cv2.imwrite(out_path,img)


if __name__=="__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='Signature Augmentation')
    # parser.add_argument("--output_folder", type=str, required=True)
    # parser.add_argument("--input_signature", type=str, required=True)
    # parser.add_argument("--svd_kept_feat", type=int, default=30, required=False)
    # parser.add_argument("--erosion_blur_rate", type=int, default=10, required=False)
    # parser.add_argument("--erosion_kernel_size", type=int, default=1, required=False)   

    # args = parser.parse_args()
    # main(args)
    raw_sig_path = "./raw_sig/*"
    extract_sig(raw_sig_path)