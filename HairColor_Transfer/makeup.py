import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import argparse

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
img_name=os.path.abspath(os.path.join(parent_path, 'input','img.jpg'))
cp_name=os.path.abspath(os.path.join(current_path, 'cp','79999_iter.pth'))

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default=img_name)
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    #if part == 17:
        #changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed


if __name__ == '__main__':
    # 1  face
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair

    args = parse_args()

    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }

    image_path = args.img_path
    cp = cp_name

    image = cv2.imread(image_path)
    image = cv2.resize(image,(1024,1024))
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
    
    parts = [table['hair']]

    colors = [[255, 255, 0]]

    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)

    save_name=os.path.abspath(os.path.join(parent_path, 'output')) # 기본 경로
    save_result_img=os.path.abspath(os.path.join(save_name,'HairStyleChage.jpg')) # 이미지가 저장될 경로

    if not os.path.exists(save_name): # 해당 경로가 존재하지 않는다면
     os.makedirs(save_name)
  
    cv2.imwrite(save_result_img, image) # 이미지 저장
  
    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    cv2.imshow('color', cv2.resize(image, (512, 512)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()















