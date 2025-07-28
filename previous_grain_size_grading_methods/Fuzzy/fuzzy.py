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

def FuzzyInterface1(I,t):
    L = len(I)
    n = 28
    greyImage = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # Change the grayscale image into a array
    I = np.asarray(greyImage, dtype="int32")

    # Find the Gradient

    for i in range(math.ceil((n-1)/2),L-math.ceil((n-1)/2)):
        for j in range(math.ceil((n-1)/2),L-math.ceil((n-1)/2)):
            a = 0
            for l in range(n):
                for k in range(n):
                    a += I[l,k]
            a = a/n**2
            u = 1/(math.exp(I[i][j]-a)+1)
            if u >= t:
                I[i][j] = 0
            else:
                I[i][j] = 255

    return I

def fuzzy(I,O):
    image = O.convert("L")
    t = otsu(image) / 255
    zhongzhi = cv2.blur(I, (3, 3))
    gaosi = cv2.GaussianBlur(zhongzhi, (3, 3), 1)
    out = FuzzyInterface1(gaosi, t)
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


    image = cv2.imread('./test_picture/108-103-6.5-500x_2.jpg')
    image_o = Image.open('./test_picture/108-103-6.5-500x_2.jpg')
    intercept_image = fuzzy(image,image_o)
    cv2.imwrite('./test_picture/108-103-6.5-500x_2_Fuzzy_final.jpg', intercept_image)
