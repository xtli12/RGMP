import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

def get_convblock(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batchnorm=True):
    conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding, stride=stride)
    if batchnorm:
        layers = [conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers) 

class WaveletCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=14, init_weights=True, prelayers=False):
        super(WaveletCNN, self).__init__()
        self.in_channels = in_channels

        self.dwt1 = DWTForward(J=1, mode='zero', wave='haar')
        self.dwt2 = DWTForward(J=1, mode='zero', wave='haar')
        self.dwt3 = DWTForward(J=1, mode='zero', wave='haar')
        self.dwt4 = DWTForward(J=1, mode='zero', wave='haar')

        self.prelayers = prelayers
        if prelayers:
            self.conv0   = get_convblock(in_channels,64,kernel_size=3)
            self.conv0_2 = get_convblock(64,64,kernel_size=3,stride=2)

        # k1
        self.conv1   = get_convblock(in_channels*4+64*prelayers,64,kernel_size=3)
        self.conv1_2 = get_convblock(64,64,kernel_size=3,stride=2)

        # do the projections include relu (and batchnorm)? (inception does include bn and relu)
        #channels1x1 = 64
        self.conv_a = get_convblock(in_channels*4,64)
        # self.conv_proj_a = nn.Conv2d(in_channels*3*4,channels1x1,kernel_size=1)
        # self.proj1  = nn.Conv2d(in_channels,channels1x1,kernel_size=1, stride=2)
        self.conv_proj_a = get_convblock(in_channels*4,64,kernel_size=1,padding=0)
        self.proj1  = get_convblock(in_channels*3,64,kernel_size=1, stride=2,padding=0)

        # k2
        self.conv2   = get_convblock(128+2*64,128,kernel_size=3)
        self.conv2_2 = get_convblock(128,128,kernel_size=3,stride=2)

        self.conv_a2   = get_convblock(in_channels*4,64)
        self.conv_a2_2 = get_convblock(64,128)
        #self.conv_proj_a2 = nn.Conv2d(in_channels*3*3*4,channels1x1,kernel_size=1)
        #self.proj2  = nn.Conv2d(128+2*channels1x1,channels1x1,kernel_size=1, stride=2)
        self.conv_proj_a2 = get_convblock(in_channels*4,128,kernel_size=1,padding=0)
        self.proj2  = get_convblock(128+2*64,128,kernel_size=1, stride=2,padding=0)

        # k3
        self.conv3   = get_convblock(2*128+2*128,256,kernel_size=3)
        self.conv3_2 = get_convblock(256,256,kernel_size=3,stride=2)

        self.conv_a3   = get_convblock(in_channels*4,64)
        self.conv_a3_2 = get_convblock(64,128)
        self.conv_a3_3 = get_convblock(128,256)
        # self.conv_proj_a3 = nn.Conv2d(in_channels*3*3*3*4,channels1x1,kernel_size=1)
        # self.proj3  = nn.Conv2d(2*128+2*channels1x1,channels1x1,kernel_size=1, stride=2)
        self.conv_proj_a3 = get_convblock(in_channels*4,128,kernel_size=1,padding=0)
        self.proj3  = get_convblock(2*128+2*128,128,kernel_size=1, stride=2,padding=0)

        # k3
        self.conv4   = get_convblock(2*256+2*128,512,kernel_size=3)
        self.conv4_2 = get_convblock(512,512,kernel_size=3,stride=2)
        #self.proj4  = nn.Conv2d(2*256+2*channels1x1,channels1x1,kernel_size=1, stride=2)
        self.proj4  = get_convblock(2*256+2*128,256,kernel_size=1, stride=2,padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear((512+256) * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        kl1, kh1_ = self.dwt1(x)
        kh1_s = kh1_[0].size()
        kh1 = kh1_[0].view(kh1_s[0],kh1_s[1]*kh1_s[2],kh1_s[3],kh1_s[4])
        kl2, kh2_ = self.dwt2(kl1)
        kh2_s = kh2_[0].size()
        kh2 = kh2_[0].view(kh2_s[0],kh2_s[1]*kh2_s[2],kh2_s[3],kh2_s[4])
        kl3, kh3_ = self.dwt3(kl2)
        kh3_s = kh3_[0].size()
        kh3 = kh3_[0].view(kh3_s[0],kh3_s[1]*kh3_s[2],kh3_s[3],kh3_s[4])
        kl4, kh4_ = self.dwt4(kl3)
        kh4_s = kh4_[0].size()
        kh4 = kh4_[0].view(kh4_s[0],kh4_s[1]*kh4_s[2],kh4_s[3],kh4_s[4])

        # kh1, kl1_ = self.dwt1(x)
        # kl1_s = kl1_[0].size()
        # kl1 = kl1_[0].view(kl1_s[0],kl1_s[1]*kl1_s[2],kl1_s[3],kl1_s[4])
        # kh2, kl2_ = self.dwt2(kl1)
        # kl2_s = kl2_[0].size()
        # kl2 = kl2_[0].view(kl2_s[0],kl2_s[1]*kl2_s[2],kl2_s[3],kl2_s[4])
        # kh3, kl3_ = self.dwt3(kl2)
        # kl3_s = kl3_[0].size()
        # kl3 = kl3_[0].view(kl3_s[0],kl3_s[1]*kl3_s[2],kl3_s[3],kl3_s[4])
        # kh4, kl4_ = self.dwt4(kl3)
        # kl4_s = kl4_[0].size()
        # kl4 = kl4_[0].view(kl4_s[0],kl4_s[1]*kl4_s[2],kl4_s[3],kl4_s[4])

        if self.prelayers:
            c0   = self.conv0(x)
            c0_2 = self.conv0_2(c0)

        # first main branch block (k1)
        if self.prelayers:
            c1_inputs = torch.hstack([kh1,kl1,c0_2])
        else:
            c1_inputs = torch.hstack([kh1,kl1])

        c1 = self.conv1(c1_inputs)
        c1_2 = self.conv1_2(c1)

        ca = self.conv_a(torch.hstack([kh2,kl2])) # 3x3 conv of 2nd level dwt
        cpa = self.conv_proj_a(torch.hstack([kh2,kl2]))# 1x1 conv of 2nd level dwt
        p1 = self.proj1(kh1) # projection of input

        x2 = torch.hstack([p1,c1_2,ca,cpa])

        # k2
        c2 = self.conv2(x2)
        c2_2 = self.conv2_2(c2)

        ca2   = self.conv_a2(torch.hstack([kh3,kl3])) # 3x3 conv of 3nd level dwt
        ca2_2 = self.conv_a2_2(ca2)
        cpa2 = self.conv_proj_a2(torch.hstack([kh3,kl3]))# 1x1 conv of 3nd level dwt
        p2 = self.proj2(x2) # projection of input

        x3 = torch.hstack([p2,c2_2,ca2_2,cpa2])

        # k3
        c3 = self.conv3(x3)
        c3_2 = self.conv3_2(c3)

        ca3   = self.conv_a3(torch.hstack([kh4,kl4])) # 3x3 conv of 3nd level dwt
        ca3_2 = self.conv_a3_2(ca3)
        ca3_3 = self.conv_a3_3(ca3_2)
        cpa3 = self.conv_proj_a3(torch.hstack([kh4,kl4]))# 1x1 conv of 3nd level dwt
        p3 = self.proj3(x3) # projection of input

        x4 = torch.hstack([p3,c3_2,ca3_3,cpa3])

        c4 = self.conv4(x4)
        c4_2 = self.conv4_2(c4)
        p4 = self.proj4(x4)

        x5 = torch.hstack([p4,c4_2])
 
        avg_p = self.avgpool(x5)

        return self.classifier(avg_p.flatten(1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
