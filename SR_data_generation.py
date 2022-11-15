!pip install georasters
import gdal
import georaster as gr
from matplotlib import image
from matplotlib import colors
import glob
import numpy as np
import os
from tqdm import tqdm
import imageio
import rasterio
import argparse
from rasterio.plot import reshape_as_image
from skimage import img_as_ubyte
import random
import PIL
import cv2
from PIL import Image


#create list of crops with coordinates Top-Left
def select_crops(initial_width,initial_height,target_size=384):
    crop_rect = []
    initial_width = initial_width - target_size
    initial_height = initial_height - target_size
    for x in range(0,initial_width,target_size):
        for y in range(0,initial_height,target_size):
            crop_rect.append({"left": x, "top": y})
    print("total number of crops= ", len(crop_rect))
    return crop_rect


def convert_s2(input_path, output_path, crop_rect, target_size, bands, rgb_mode=False,s2_file_name=None):
    lr_shape=(96, 96, 3)
    with rasterio.open(input_path) as s2_data:
        s2 = s2_data.read(bands)
    cat_image = reshape_as_image(s2)

    if s2_file_name is None :
        file_name = input_path
    else:
        file_name = s2_file_name
    file_name = file_name.split("/")[-1].split(".")[0]
    file_name = os.path.join(output_path, file_name)

    for index in range(len(crop_rect)):
        left = crop_rect[index]["left"]
        top = crop_rect[index]["top"]
        right = left + target_size
        bottom = top + target_size
        new_image = cat_image[top:bottom, left:right]
        new_image = np.moveaxis(new_image, -1, 0)
        new_file_name = file_name + "_" + str(index) + ".tif"
        if rgb_mode:
            display_channels = np.array(bands) - 2
            display_channels = display_channels.tolist()
            s2 = new_image.astype(np.float32)
            min_value, max_value = np.min(s2), np.max(s2)
            s2 = np.clip(s2, min_value, max_value)
            s2 /= max_value
            s2 = s2.astype(np.float32)
            #s2 = s2[display_channels, :, :]
            rgb = np.rollaxis(s2, 0, 3)
            imageio.imsave(new_file_name.replace('tif', 'png'), img_as_ubyte(rgb))

        else:
            with rasterio.open(new_file_name, 'w',width=target_size, height=target_size,
                               count=len(bands),driver="Gtiff", dtype=rasterio.uint16) as dst:
                dst.write(new_image)


def cut_sen2_files(target_size=384):
    # path of original hr images
    path = 'D:/work/'
    #save high resolution data after cropping here
    output_path = 'C:/Users/Mohammad Ramadan/Desktop/data/SuperResolution/output/'
    for p in os.listdir(path):
        f= gr.from_file(path+p)
        file_name = p

        ch,initial_height, initial_width = f.shape
        crop_rect = select_crops(initial_width, initial_height, target_size)

        bands = [5, 3, 2]
        convert_s2(path+file_name, output_path ,crop_rect, target_size, bands,
                        rgb_mode=True)


cut_sen2_files()

path_hr = 'C:/Users/Mohammad Ramadan/Desktop/data/SuperResolution/output/'
#path of hr data after cleaning
path_hr_cropped = 'C:/Users/Mohammad Ramadan/Desktop/data/SuperResolution/outputhr_cropped/'
path_hr_processed = 'C:/Users/Mohammad Ramadan/Desktop/data/SuperResolution/outputhr_processed/'
hr_shape=(384, 384, 3)
for p in os.listdir(path_hr):
  img = cv2.imread(path_hr+p)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
  contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnt = contours[0]
  x,y,w,h = cv2.boundingRect(cnt)
  crop = img[y:y+h,x:x+w]
  img = Image.fromarray(crop)
  img = img.resize((hr_shape[0],hr_shape[1]),PIL.Image.NEAREST)
  img.save(path_hr_cropped+p, 'png')

for p in os.listdir(path_hr_cropped):
  # read image
  img = cv2.imread(path_hr_cropped+p)
  # blur
  blur = cv2.GaussianBlur(img, (3,3), 0)
  # convert to hsv and get saturation channel
  sat = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)[:,:,1]
  # threshold saturation channel
  thresh = cv2.threshold(sat, 50, 255, cv2.THRESH_BINARY)[1]
  # apply morphology close and open to make mask
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
  morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
  mask = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
  # do OTSU threshold to get image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  # write black to otsu image where mask is black
  otsu_result = otsu.copy()
  otsu_result[mask==0] = 0
  # write black to input image where mask is black
  img_result = img.copy()
  img_result[mask==0] = 0
  img_final = Image.fromarray(img_result)
  img_final.save(path_hr_processed+p, 'png')

lr_shape=(96, 96, 3)
outputpath = 'C:/Users/Mohammad Ramadan/Desktop/data/SuperResolution/input/'
for p in os.listdir(path_hr_processed):
  print(p)
  image = Image.open(path_hr_processed+p) 
  lr_resized=np.array(image.resize((lr_shape[0],lr_shape[1]),PIL.Image.BICUBIC))
  img = Image.fromarray(lr_resized)
  img.save(outputpath+p, 'png')

