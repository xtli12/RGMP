import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MINSH(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        x = x * (torch.tanh(F.softplus(x)))
        return x

class _BM(nn.Sequential):
    def __init__(self, num_input_features):
        super(_BM, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('mish1', MINSH()),

    def forward(self, x):
        new_features = super(_BM, self).forward(x)

        return new_features


class _ResLayer(nn.Sequential):
    def __init__(self, num_input_features, out_feature):
        super(_ResLayer, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, out_feature, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_feature)
        self.mish = MINSH()

        self.conv2 = nn.Conv2d(out_feature, num_input_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(num_input_features)

    def forward(self, x):
        x1 = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.mish(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.mish(x)
        x = x1 +x

        return x

class CSP(nn.Module):
    def __init__(self, num_input_features,output_feature):
        super(CSP,self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_input_features, kernel_size=3, stride=1,padding=1, bias=False)
        self.cbm1 = _BM(num_input_features)
        self.route = nn.Conv2d(num_input_features, output_feature, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(output_feature, output_feature, kernel_size=3, stride=1, padding=1,bias=False)
        self.cbm2 = _BM(output_feature)
        self.conv3 = nn.Conv2d(output_feature, output_feature, kernel_size=3, stride=1, padding=1,bias=False)
        self.cbm3 = _BM(output_feature)
        self.conv4 = nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1, bias=False)
        self.cbm4 = _BM(num_input_features)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.cbm1(x1)
        x = self.route(x1)
        x2 = self.conv2(x)
        x2 = self.cbm2(x2)
        x3 = self.conv3(x2)
        x3 = self.cbm3(x3)
        x4 = torch.cat([x2, x3], 1)
        x4 = self.conv4(x4)
        x4 = self.cbm4(x4)
        out = torch.cat([x1, x4], 1)

        return out



class fic(nn.Module):
    def __init__(self, num_classes = 14):
        super().__init__()

        self.classifier0 = nn.Linear(512, num_classes)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.cbm2 = _BM(64)
        self.res1 = _ResLayer(64,32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.cbm3 = _BM(128)
        self.res2 = _ResLayer(128,64)
        self.res3 = _ResLayer(128,64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.csp1 =CSP(128,64)
        self.csp2 =CSP(256,128)
        self.csp3 =CSP(512,256)
        self.conv4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.cbm4 = _BM(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.cbm5 = _BM(512)
        self.conv6 = nn.Conv2d(512,256, kernel_size=1, stride=1, bias=False)
        self.cbm6 = _BM(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.cbm7 = _BM(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbm2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.cbm3(x)
        x = self.res2(x)
        x= self.res3(x)
        x = self.maxpool(x)
        x = self.csp1(x)
        x = self.maxpool(x)
        x = self.csp2(x)
        x = self.maxpool(x)
        x = self.csp3(x)
        x = self.conv4(x)
        x = self.cbm4(x)
        x = self.conv5(x)
        x = self.cbm5(x)
        x = self.conv6(x)
        x = self.cbm6(x)
        x = self.conv7(x)
        x = self.cbm7(x)
        out = x.mean(dim=[2, 3])
        out = self.classifier0(out)
        return out
if __name__ == '__main__':
    batch_size = 1
    net = fic()
    # x = torch.rand(batch_size, 3,1376,1104)
    x = torch.rand(batch_size, 3, 224, 224)
    print(1)
    print(net(x).size())
