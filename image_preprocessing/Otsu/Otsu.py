import math 
from torchvision import *
from fuzzy_logical import *
import cv2
from Otsu1 import *


def ObtainGradient(inputImage):
    # Convert image to grayscale
    greyImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    # Change the grayscale image into a array
    imageArray = np.asarray(greyImage, dtype="int32")

    # Find the Gradient
    I = np.gradient(imageArray)

    return I

def Interface1(I,t):
    L = len(I)
    greyImage = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # Change the grayscale image into a array
    I = np.asarray(greyImage, dtype="int32")

    # Find the Gradient

    for i in range(L):
        for j in range(L):
            if I[i][j] >= t:
                I[i][j] = 255
            else:
                I[i][j] = 0

    return I

def Otsu(I,O):
    image = O.convert("L")
    t = otsu(image)
    zhongzhi = cv2.blur(I, (3, 3))
    gaosi = cv2.GaussianBlur(zhongzhi, (3, 3), 1)
    out = Interface1(gaosi, t)
    out = torch.tensor(out)
    out = out.unsqueeze(0)
    e = out
    c = out
    intercept_image = torch.cat([out, e, c], 0)
    intercept_image = intercept_image.transpose(2, 0)
    intercept_image = intercept_image.transpose(1, 0)
    intercept_image = intercept_image.data.numpy()

    return intercept_image

if __name__ == "__main__":


    image = cv2.imread('./test_picture/51276.jpg')
    image_o = Image.open('./test_picture/51276.jpg')
    intercept_image = Otsu(image,image_o)
    cv2.imwrite('./test_picture/51276_otsu.jpg', intercept_image)  # 保存这个图片
