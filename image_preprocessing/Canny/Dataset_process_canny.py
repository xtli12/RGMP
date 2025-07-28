import os
import os.path 
import glob
import numpy as np
import argparse
from torchvision import *
from PIL import Image
import cv2
import numpy as np
# import pylab as pl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.signal
from FWLBP import *
from skimage.feature import local_binary_pattern

from Fuzzy_origianl_test import *


parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--data_path", type=str, default='./datasets/train_rm_dup_rmerror/', help="data path")
parser.add_argument("--train_data_path1", type=str, default='./datasets/train_rm_dup_rmerror_Canny/',help="preprocess_data_path")
parser.add_argument("--train_data_path2", type=str, default='./train_rm_dup_rmerror_newep24_addLBP/',help="preprocess_data_path")
opt = parser.parse_args()




def main():
    data = data_augment(opt.data_path, opt.train_data_path1)
    data.augment()



class data_augment():
    def __init__(self,data_path,output_path):
        self.data = data_path
        self.out = output_path


    # convert gray to rgb image
    def Canny(slfe,img):
        v1 = cv2.Canny(img, 60, 90)
        v1 = torch.Tensor(v1)
        v1 = v1.unsqueeze(0)
        a = torch.cat([v1, v1, v1], 0)
        a = a.transpose(2, 0)
        a = a.transpose(1, 0)
        a = a.numpy() 
        return a

    def augment(self):
        b = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]  # 晶粒度等级
        # b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for k in range(14):

            a1 = str(b[k])  
            img_path = os.path.join(self.data, a1)  
            img_path_name = glob.glob(os.path.join(img_path, '*.jpg'))
            img_path_name.sort()  
            result_path = os.path.join(self.out, a1)  
            if not os.path.exists(result_path): 
                os.makedirs(result_path)  
            for i in range(len(img_path_name)):
                # load rgb images
                print(img_path_name[i])
                d = img_path_name[i].split('/')[-1][0:18]  

                image = cv2.imread(img_path_name[i])
                a = self.Canny(image)
                a = 255 - a  
                rgb_name = d  + '_14'  
                rgb_dir = os.path.join(result_path, rgb_name) 
                rgb_dir =rgb_dir + '.jpg'
                cv2.imwrite(rgb_dir , a)  

            print(b[k])



if __name__ == '__main__':
    main()
    print('Get it!')
