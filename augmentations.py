import cv2
import random
import numpy as np

#this function changes the brightness of an image
def change_brightness(img):
    brightness = random.randint(-30,30)
    print("the brightness value is: ", brightness)
    out = cv2.addWeighted( img, 1, img, 0, brightness)
    return out


#this function changes the saturation of an image
def adjust_saturation(img):
    # Convert BGR image to HSV
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imghsv)
    random_factor =(random.randint(-20,20)/100) + 1
    s = random_factor * s
    print("the saturation value is: ", random_factor)
    s[s>255] = 255 #set the value to 255 if it is above 255 (any value above 255 is not valid)
    s = np.array(s, dtype='uint8')
    imghsv = cv2.merge([h, s, v])
    imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return imgrgb


#this function changes the color of an image by adding a random value to one of the channels
def change_color(image):
    choices = [ 0 , 1 , 2 ]
    random_channel = random.choice(choices)
    random_values = random.randint(130,134) * 0.01
    added_weight = image[:,:, random_channel]  * random_values
    added_weight[added_weight>255] = 255
    added_weight = np.array(added_weight, dtype='uint8')
    image[:, :, random_channel] = added_weight
    return image


#this function applies histogram equalization to an image
#CLAHE is a contrast enhancement algorithm that is used to improve the contrast of an image
def hisogram_equlization(image):
    global CLAHE
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape)>2:
        l,a,b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        l = CLAHE.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    else:
        return CLAHE.apply(image)
    

if __name__ == "__main__":
    images_path = r"path/to/image"
    img = cv2.imread(images_path)
    hisogram_equlization_applied = hisogram_equlization(img)
    cv2.imwrite("hisogram_equlization_applied.bmp", hisogram_equlization_applied)

