import os 
import random 
import glob 
import multiprocessing 
import random 
import argparse
import matplotlib.pyplot as plt

import cv2
import numpy as np 
from PIL import Image
import utils
from tqdm import tqdm
import time

from dev import augment_2, read_gray


def random_color_signature():
    p_color = random.uniform(0,1)
    if p_color < 0.5:
        color = 'red'
    elif p_color < 0.8:
        color = 'blue'
    else:
        color = 'black'
    return color


def random_coordinate(bg_shape, im_shape):
    bg_w, bg_h = bg_shape
    im_w, im_h = im_shape
    x = random.randint(0, bg_w-im_w)
    y = random.randint(0, bg_h-im_h)
    return x,y


def gen(ind):
    # import ipdb; ipdb.set_trace()
    label_yolo = []

    # Number of signatures in background
    num_signs = random.randint(2, 5)

    # Random background
    background = Image.open(random.choice(list_bgs))
    bg_w, bg_h = background.size
    short_size = min(bg_h, bg_w)

    # Random symbol
    symbol = Image.open(random.choice(list_symbols)).convert("RGBA")
    sym_w = int(random.uniform(0.05, 0.3) * bg_w)
    sym_h = int(random.uniform(0.05, 0.3) * bg_h)
    symbol = symbol.resize((sym_w, sym_h))
    #symbol = utils.create_transparent_image(symbol)
    # Random coordinate or do not need
    sym_x, sym_y = random_coordinate(background.size, symbol.size)
    # Fill
    background = utils.overlay_transparent(background=background,\
                        foreground=symbol, coordinate=(sym_x, sym_y), remove_bg=False)['filled_image']

    # Random stamp
    stamp = Image.open(random.choice(list_stamps))
    stp_w, stp_h = stamp.size
    ratio_size = random.uniform(0.05, 0.25) 
    ratio_stp = stp_h / stp_w
    new_stp_w = int(ratio_size * short_size)
    new_stp_h = int(ratio_stp * new_stp_w)
    stamp = stamp.resize((new_stp_w, new_stp_h))
    stamp = utils.create_transparent_image(stamp)

    # Random coordinate or do not need
    stp_x, stp_y = random_coordinate(background.size, stamp.size)
    background = utils.overlay_transparent(background=background,\
                        foreground=stamp, coordinate=(stp_x, stp_y), remove_bg=False)['filled_image']
    
    # Saved inserted signature coordinate
    old_crd_signs = []
    # Fill signature to background
    for i in range(num_signs):
        signature = read_gray(random.choice(list_signs))
        sign_h, sign_w = signature.shape
        ratio_size = random.uniform(0.1, 0.25)
        ratio_sign = sign_h / sign_w
        new_sign_w = int(ratio_size * short_size)
        new_sign_h = int(ratio_sign * new_sign_w)
        signature = cv2.resize(signature,(new_sign_w, new_sign_h))
        
        color = random_color_signature()
        signature = augment_2(signature, color, 
                              blur_rate=args.blur_rate, 
                              erosion_kernel_size=args.erosion_kernel_size,
                              num_eigenvalues=args.num_eigenvalues) * 255
        signature = np.array(signature).astype(np.uint8)

        signature = Image.fromarray(signature).convert("RGB")

        if signature.mode != "RGBA":
            print("Not RGBA. Converting...")
            signature = utils.create_transparent_image(signature)
        else:
            print("RGBA")
        
        while True:
            # Random coordinate
            x1, y1 = random_coordinate(background.size, signature.size) 
            x2, y2 = x1+new_sign_w, y1+new_sign_h
            check = 0
            for coord in old_crd_signs:
                x1_old, y1_old, w_old, h_old = coord
                x2_old, y2_old = x1_old+w_old, y1_old+h_old
                # Check if overlapped signatures
                check += ((x1 in range(x1_old, x2_old) or x2 in range(x1_old, x2_old)) and y1 in range(y1_old, y2_old)) \
                        or ((x1 in range(x1_old, x2_old) or x2 in range(x1_old, x2_old)) and y2 in range(y1_old, y2_old)) \
                        or ((x1_old in range(x1, x2) or x2_old in range(x1, x2)) and y1_old in range(y1, y2)) \
                        or ((x1_old in range(x1, x2) or x2_old in range(x1, x2)) and y2_old in range(y1, y2))
                if check != 0:
                    break

            if check==0:
                break
        
        old_crd_signs.append((x1, y1, new_sign_w, new_sign_h))
        # random add stamp under signature
        if random.uniform(0, 1) < 0.1:
            off_x, off_y = random.randint(0, new_sign_w), random.randint(0, new_sign_h) 
            # Fill stamp
            background = utils.overlay_transparent(background=background, \
                            foreground=stamp, coordinate=(max(0, x1-off_x), max(0, y1-off_y)))['filled_image']
            # Fill signature                                    
            background = utils.overlay_transparent(background=background, \
                            foreground=signature, coordinate=(x1, y1),remove_bg=False)['filled_image']
        else:
            if random.uniform(0, 1) < 0.1:
                background = utils.overlay_transparent(background=background, \
                                foreground=signature, coordinate=(x1, y1), remove_bg=False)['filled_image']
            else:
                # Remove background
                background = utils.overlay_transparent(background=background, \
                                foreground=signature, coordinate=(x1, y1), remove_bg=False)['filled_image']

        # Get label yolo format
        x_c, y_c, w_c, h_c = utils.tlwh_2_yolo_format((x1, y1, new_sign_w, new_sign_h), (bg_w, bg_h))
        stamp_label = "{} {} {} {} {}".format(STAMP, x_c, y_c, w_c, h_c) + "\n"
        label_yolo.append(stamp_label)
    
    # Save image and label
    background.save(os.path.join(args.output_folder, "signature_%d.jpg" %ind))
    # background.show()
    with open(os.path.join(args.output_folder, "signature_%d.txt" %ind), "w") as f:
        f.writelines(label_yolo)


def main(args):
    os.makedirs(args.output_folder,exist_ok=True)
    
    global list_symbols
    global list_signs
    global list_stamps
    global list_bgs
    global STAMP
    STAMP = 0

    list_symbols = glob.glob(os.path.join(args.input_symbol, "*"))
    list_signs = glob.glob(os.path.join(args.input_signature,"*"))
    list_stamps = glob.glob(os.path.join(args.input_stamp, "*"))
    list_bgs = glob.glob(os.path.join(args.input_background, "*"))

    pool = multiprocessing.Pool(args.num_workers)
    output = list(tqdm(
        pool.imap(gen, range(args.num_sample)), total=args.num_sample, desc="Augmenting"))
    pool.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Signature Augmentation')
    parser.add_argument('--output_folder', default='./result/', 
                        type=str, required=False, help='folder path to output datasets')
    parser.add_argument('--input_signature', type=str, required=True, 
                        help='folder path to input signature')
    parser.add_argument('--input_symbol', type=str, required=True, 
                        help='folder path to input other symbols')
    parser.add_argument('--input_stamp', type=str, required=True, 
                        help='folder path to input other signature')
    parser.add_argument('--input_background', type=str, required=True, 
                        help='folder path to input background')                   
    parser.add_argument('--blur_rate', type=int, required=False,
                        help='blur rate in augmentation')
    parser.add_argument('--erosion_kernel_size', type=int, required=False,
                        help='erosion kernel size in augmentation')
    parser.add_argument('--num_eigenvalues', type=int, required=False,
                        help='number of kept eigenvalues in truncated svd in augmentation')
    parser.add_argument('--num_sample', default= 100, 
                        type=int, required=False, help='number of image')
    parser.add_argument('--num_workers', default=8, type=int, 
                        required=False, help='number of core use for multiprocessing')

    global args
    args = parser.parse_args()
    main(args)