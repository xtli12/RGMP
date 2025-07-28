# from swin_Den4KVStage3q_sota_visualize import *
import argparse 
import torch.optim as optim 
import torch.backends.cudnn as cudnn 
from torch.utils.data import DataLoader 
from torch.autograd import Variable 
import os 
import time 
from utils_swin import AverageMeter, initialize_logger, save_checkpoint, record_loss 
import torchvision
from torchvision import transforms
import shutil
import cv2
from PIL import Image
# from swin1_simple_visualize import *
# from swin_Den4KVStage3q_sota_visualize import *
from swin_Den4KVStage3q_sota_densenet_visualize import *


# model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24))
model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0)

# resume_file = os.path.join(os.path.join('./Results/GSNet_visualize/'), 'net_14epoch.pth') 
resume_file = os.path.join(os.path.join('./Results/GSNet_visualize/'), 'net_417epoch.pth') 
if resume_file:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda(0))
        # start_epoch = checkpoint['epoch']
        # iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['model'], strict = False)
        # optimizer.load_state_dict(checkpoint['optimizer'])


# image_o = Image.open('./test_picture/pic_in_ppt.jpg')
image_o = Image.open('./test_picture/Fill.jpg')
resize = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])
# t = transforms.ToTensor()
x = resize(image_o)
for j in range(3):
    w = x[j:j+1, :,: ]
    w_detached = w.detach().numpy()
    det_B = np.linalg.det(w_detached)
    if det_B == 0:
        print( det_B)
        print(w*255)

# x = torch.tensor(x)
x = x.unsqueeze(0)
before,after = model(x)
before = before.squeeze(0)
after = after.squeeze(0)

for i in range(736):
# for i in range(1024):
# for i in range(1526):
    v = after[i:i+1, :,: ]
    v_detached = v.detach().numpy()
    det_A = np.linalg.det(v_detached)
    if det_A != 0:
        print((v+1)*255/2)
        AAA = (v+1)*255/2
        AA = AAA.detach().numpy()
        AA= np.linalg.det(AA)
        print("行列式是", AA)
        v = torch.cat([v, v, v], 0)
        v = v.transpose(2, 0)
        v = v.transpose(1, 0)
        v = v.data.numpy()
        v = v*255


        a = np.zeros((56, 56, 3))
        b = np.zeros((56, 56, 3)) + 255
        a[:,:,0] = b[:, :,0];a[:,:,1] = b[:, :,0];
        a[:,:,2] = v[:, :,2]

        a = np.uint8(a)
        a = cv2.applyColorMap(a, cv2.COLORMAP_JET)

        result_path = './test_picture/Fill/channel_'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        cv2.imwrite(result_path + str(i) + '.jpg', a)





