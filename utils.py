import numpy as np 
import os 
import cv2 
from PIL import Image, ImageFont, ImageFont
import PIL
import random

def tlwh_to_polygon(left, top, width, height):
    '''Convert bounding box (left, top, width, height) to polygon 4 point
    '''
    x, y = left, top
    return [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height)
    ]


def get_fontsize(font_path, text, max_width, max_height):
    ''' Get fontsize fit bounding box - PIL
    '''
    fontsize = 1
    font = ImageFont.truetype(font_path, fontsize) 
    while font.getsize(text)[1] < max_height and font.getsize(text)[0] < max_width:
        fontsize += 1
        font = ImageFont.truetype(font_path, fontsize)
    return fontsize


def get_coord_text(font, text, xy):
    ''' Get bounding box fit mast text - PIL
    Return
    ------
    :Bounding box: (left, top, width, height)
    '''
    first=True
    left_text = None
    top_text = None
    right_text = None
    bot_text = None

    for k, char in enumerate(text):
        if char == " ":
            continue
        # Get coordinate of each letters
        bottom_1 = font.getsize(text[k])[1]
        right, bottom_2 = font.getsize(text[:k+1])
        bottom = bottom_1 if bottom_1<bottom_2 else bottom_2
        width_char, height_char = font.getmask(char).size
        right += xy[0]
        bottom += xy[1]
        top = bottom - height_char
        left = right - width_char
        # Get coordinate of the first letter
        if first:
            left_text = left
            top_text = top
            first = False

        right_text = right
        bot_text = bottom

    if None in [left_text, right_text, bot_text, top_text]:
        raise ValueError("Invalid coordinate of the text. \
                        Expect 4 numbers but get None!")
        # return (xy[0], xy[1], 0, 0)
    width_text = right_text - left_text
    height_text = bot_text - top_text

    return (left_text, top_text, width_text, height_text)


def overlay_huge_transparent(background:PIL.Image, foreground:PIL.Image, color=None):
    ''' Overlay huge transparent image on background
    Params
    ------
    :background: PIL.Image 
    :foreground: PIL.Image - RGBA image

    Returns
    -------
    :image: PIL.Image
    '''
    
    bg_w, bg_h = background.size

    cur_fore_h, cur_fore_w = foreground.size
    if random.uniform(0, 1) < 0.5:  # big
        ratio = random.uniform(3, 4)
    else:
        ratio = random.uniform(0.9, 2)

    new_fore_h = int(ratio * bg_h)
    new_fore_w = int((new_fore_h/cur_fore_h) * cur_fore_w)
    foreground = foreground.resize((new_fore_w, new_fore_h))    

    x = random.randint(int(bg_w/2 - new_fore_w), int(bg_w/2))
    y = random.randint(int(bg_h/2 - new_fore_h), int(bg_h/2))

    background = overlay_transparent(background=background, foreground=foreground,  \
                                    coordinate=(x,y), color=color)['filled_image']

    return background


def PIL_to_gray(img:PIL.Image):
    temp = np.array(img)
    if np.mean(2*temp[:,:,0]-temp[:,:,1]-temp[:,:,2]) != 0:
        raise Exception("Input is not 'gray' enough... IT IS NOT GRAY AT ALL!!!")
    else:
        return temp[:,:,0]


def overlay_transparent(background:PIL.Image, foreground:PIL.Image, coordinate=None, remove_bg=False, color=None):
    ''' Overlay transparent image on background
    Params
    ------
    :background: PIL.Image 
    :foreground: PIL.Image - RGBA image
    :coordinate: list(x,y) - Left-top coordinate on background
    :remove_bg: bool - remove background of inserted image

    Returns
    -------
    :image: PIL.Image
    '''
    bg_w, bg_h = background.size 
    fg_w, fg_h = foreground.size
    # Left-top coordinate of inserted image
    if coordinate is None:
        x = random.randint(0, bg_w - fg_w)
        y = random.randint(0, bg_h - fg_h)
    else:
        x, y = coordinate

    wid = fg_w
    hei = fg_h
    

        
    # If remove background if inserted image
    if remove_bg:
        inserted_area = np.array(background)[y:y+hei, x:x+wid, :]
        dominant_color = find_dominant_color(inserted_area)
        background = np.array(background)
        background[y:y+hei, x:x+wid, :] = dominant_color
        background = Image.fromarray(background)
    
    # Change color of foreground
    if color is not None:
        foreground = change_color_transparent(foreground, color)
    
    if foreground.mode != "RGBA":
        print("Bad transparent foreground")
        foreground = foreground.convert("RGBA")
    # Overlay transparent
    background.paste(foreground, (x,y), foreground)
    
    return {
        "filled_image":background,
        "bbox": (x,y, wid, hei)
    }


def change_color_transparent(image, color=(0,0,0)):
    image = np.array(image)
    try:
        image[:,:,0] = color[0]
        image[:,:,1] = color[1]
        image[:,:,2] = color[2]
        image = Image.fromarray(image)
    except:
        print(image.shape)
    return image


def create_transparent_image(image, threshold=225):
    '''Create transparent image from white background image
    Params
    ------
    :image 
    :threshold: threshold to remove white background

    Return
    ------
    image: PIL.Image
    '''
    if isinstance(image, str):
        image = Image.open(image).convert("RGBA")
    else: # PIL.Image
        image = image.convert("RGBA")
    
    # Version PIL
    width, height = image.size
    pixel_data = image.load()
    # Set alpha channel to zerp pixel value > threshold
    for y in range(height):
        for x in range(width):
            if all(np.asarray(pixel_data[x,y][:3]) > threshold):
                pixel_data[x, y] = (255, 255, 255, 0)
                # print(type(pixel_data))
                # import sys; sys.exit()
            else:
                #pixel_data[x, y] = (0,0,0,255)
                temp = []
                temp.append(list(pixel_data[x,y][:3]))
                temp.append([200])
                temp = sum(temp, [])
                pixel_data[x,y] = tuple(temp)


    return image


def find_dominant_color(background: np.ndarray):
    '''find the dominant color on background
    
    Params
    ------
    :background: PIL.Image - "RGB"
    '''
    image = Image.fromarray(background)
    #Resizing parameters
    width, height = 448, 448
    # image = Image.fromarray(background)
    image = image.resize((width, height),resample = 0)
    #Get colors from image object
    pixels = image.getcolors(width * height)
    #Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    #Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    return dominant_color


def tlwh_2_yolo_format(bbox, bg_shape):
    x, y, w, h = bbox
    bg_w, bg_h = bg_shape
    x_center = np.round(((x+w/2) / bg_w), 6)
    y_center = np.round(((y+h/2) / bg_h), 6)
    width = np.round((w/bg_w), 6)
    height = np.round((h/bg_h), 6)
    return (x_center, y_center, width, height)


def gen_signature(ind):
    path = sign_images[ind]
    image = Image.open(path)
    wid, hei = image.size 
    ratio = hei/wid
    new_wid = 224
    new_hei = int(ratio * new_wid)
    image = image.resize((new_wid, new_hei))
    image = create_transparent_image(image, threshold=220)
    image.save(os.path.join("input_signature", "{}.png".format(ind)))


def gen_stamp(ind):
    path = stamp_images[ind]
    image = Image.open(path)
    image = create_transparent_image(image, threshold=220)
    image.save(os.path.join("input_stamp", "{}.png".format(ind)))


def collect_signature(path="/home/pdd/Downloads/Generated_data/Generated_data"):
    from shutil import copyfile
    import random
    temp = 0
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        img_list = [os.path.join(folder_path,img) for img in os.listdir(folder_path)]
        choosen = random.choices(img_list, k=2)
        print(choosen)
        copyfile(choosen[0], "/home/pdd/Desktop/sandbox/Fake-Data-Generator/data/signature_collection/{}.png".format(temp))
        temp += 1
        copyfile(choosen[1], "/home/pdd/Desktop/sandbox/Fake-Data-Generator/data/signature_collection/{}.png".format(temp))
        temp += 1


if __name__ == "__main__":
    from tqdm import tqdm
    import glob
    import multiprocessing
    global sign_images
    sign_images = glob.glob(os.path.join("data", "signature_collection", "*"))
    # for ind, path in tqdm(enumerate(sign_images), total=len(sign_images)):
    pool = multiprocessing.Pool(8)
    output = list(tqdm(
        pool.imap(gen_signature, range(len(sign_images))), total=len(sign_images), desc="Augmenting"))
    pool.terminate()
    pass        

    # stamp_images = glob.glob(os.path.join("data", "stamp_collection", "*"))
    # pool = multiprocessing.Pool(8)
    # output = list(tqdm(
    #     pool.imap(gen_stamp, range(len(stamp_images))), total=len(stamp_images), desc="Augmenting"))
    # pool.terminate()
    # #collect_signature()