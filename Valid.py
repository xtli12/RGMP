# -*- coding:UTF-8 -*-
# write by hyc
from PIL import Image; import torchvision
# from dense3 import 
import torch
import argparse
import os
import glob
import time
from torchvision import transforms
from GuseNet import *
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve ,auc
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


def main():
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    start_time0 = time.time()

    parser = argparse.ArgumentParser(description="SSR") 
    parser.add_argument('--img_path', type=str, default='/data/home-gxu/lxt21/new_mae/datasets/validdata_rm_dup_rmerror_ep/', help='保存路径')
    # parser.add_argument('--img_path', type=str, default='/data/home-gxu/lxt21/new_mae/datasets/validdata_metric_test/', help='保存路径')
    parser.add_argument('--model_path_GSNet', type=str,default='/data/home-gxu/lxt21/new_mae/Results/steel_gsnet_224highest_pre/net_448epoch.pth', help='pth路径')

    opt = parser.parse_args() 
    model_GSNet = GSNet (hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), growth_rate=32, block_config=(6, 12, 24, 16), window_size=7, num_init_features=64, bn_size=4, drop_rate=0)


    model_path = opt.model_path_GSNet
    model = model_GSNet
    save_point = torch.load(model_path)
    model.load_state_dict(save_point['model'])
    if torch.cuda.is_available(): 
        model.cuda()
    model.eval()

    print('--------GSNet_eval----------')

    acc = 0

    y_pred_int = []
    y_true_int = []
    y_pred_float = []
    y_true_float = []
    y_ture = []
    y_scores = []

    for k in range(14):
        ac = 0
        a = [6.5,
             7.0,
             7.5,
             8.0,
             8.5,
             9.0,
             9.5,
             10.0,
             10.5,
             11.0,
             11.5,
             12.0,
             12.5,
             13.0]
        x1 = str(a[k])
        img_path = opt.img_path + x1
        img_path_name = glob.glob(os.path.join(img_path, '*.jpg'))
        img_path_name.sort()

        for i in range(len(img_path_name)):
            rgb = Image.open(img_path_name[i])
            rgb = preimg(rgb).unsqueeze(0);
            rgb = rgb.cuda()
            output = model(rgb)
            max, pred = output.topk(1, 1, True, True)
            min, _ = output.topk(1, 1, False, True)
            original = output
            pred = pred.flatten().cpu().numpy()
            pre_label = torch.tensor([float(idx_to_class[pred[i]]) for i in range(len(pred))])
            pre_float = [float(idx_to_class[pred[i]]) for i in range(len(pred))]
            y_pred_int.append(int(pre_float[0] * 10))
            y_true_int.append(int(a[k] * 10))
            # y_pred_int_p.append(int(pre_float[0] * 10))
            # y_true_int_P.append(int(a[k] * 10))
            y_pred_float.append(pre_float[0])
            y_true_float.append(a[k])

            """softmax输出值归一化"""
            normalize_data = (original - min).true_divide(max - min)
            truth_index = list(idx_to_class.values()).index(str(a[k]))
            truth_probability = normalize_data[0, truth_index]
            truth_probability = truth_probability.cpu().detach().numpy()
            """真实样本分为正样本和负样本，一个样本输出两个概率"""

            y_ture.append(1)
            y_scores.append(truth_probability)

            y_ture.append(0)
            y_scores.append(1 - truth_probability)

            acc0, acc0_5 = accuracy(pre_label, a[k])
            # if not acc0_5 :
            #     print(img_path_name[i])
            ac += (0.4 * acc0 + 0.6 * acc0_5) / len(img_path_name)
        # p=metrics.precision_score(y_true_int_P, y_pred_int_p, average='micro')
        # print(ac)
        # print (p)
        acc += ac

    mAP1 = metrics.precision_score(y_true_int, y_pred_int, average='macro')
    print('mAP=%8f' % mAP1)

    print('准确率=%8f'%(acc/14))

    recall = metrics.recall_score(y_true_int, y_pred_int, average='micro')
    print('召回率=%8f'%(recall))

    f1 = metrics.f1_score(y_true_int, y_pred_int, average='weighted')
    print('f1_score=%8f'%(f1))
    #
    # ka = cohen_kappa_score(y_true_int, y_pred_int)
    # print('kappa score=%8f'%(ka))

    # y_true1 = np.array(y_ture)
    # y_scores = np.array(y_scores)
    # roc= roc_auc_score(y_true1, y_scores)
    # print('ROC=%8f'%(roc))
    #
    # fpr, tpr, thread = roc_curve(y_true1, y_scores)
    # roc_auc = auc(fpr, tpr)
    # # 绘图
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig('roc.png', )
    # # plt.show()
    #
    Explained = explained_variance_score(y_true_float, y_pred_float)
    print('可释方差值=%8f'%(Explained))
    #
    # Mean = mean_absolute_error(y_true_float, y_pred_float)
    # print('平均绝对误差=%8f'%(Mean))
    #
    Mean_squared = mean_squared_error(y_true_float, y_pred_float)
    print('均方误差=%8f'%(Mean_squared))
    #
    # Median = median_absolute_error(y_true_float, y_pred_float)
    # print('中值绝对误差=%8f'%(Median))
    #
    R = r2_score(y_true_float, y_pred_float)
    print('R方值，确定系数=%8f'%(R))

    #     acf.append(ac)
    #     acff += ac * len(img_path_name)
    #     numi += len(img_path_name)
    #
    # end_time0 = time.time()
    # print(end_time0 - start_time0); print(acf); print(acff/numi)

def preimg(x): # 预处理测试图像
    normalize1 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    rgb = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(),  # 压缩至640*640
         transforms.ToTensor(), normalize1])
    return rgb(x)

def accuracy(output, target):
    pre_label = output
    gt_label = target
    acc0 = torch.le(abs(pre_label - gt_label), 0)
    acc0_5 = torch.le(abs(pre_label - gt_label), 0.5)
    return acc0, acc0_5

if __name__ == '__main__':
    main() # 程序起点
    print('got it!') # 打印torch版本
