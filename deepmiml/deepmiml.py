import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN,self).__init__()
        vgg=models.vgg16(pretrained=True)
        modules=list(vgg.features[i] for i in range(29))
        self.vgg=nn.Sequential(*modules)
    def forward(self,images):
        with torch.no_grad():
            features=self.vgg(images)
        #N,C,H,W=features.size()
        #features=features.view(N,C,H*W)
        #features=features.permute(0,2,1)
        return features

class DeepMIML(nn.Module):
    def __init__(self,L=1032,K=100):
        super(DeepMIML,self).__init__()
        self.L=L
        self.K=K
        self.conv1=nn.Conv2d(in_channels=512,out_channels=L*K,kernel_size=1)
        self.pool1=nn.MaxPool2d((K,1),stride=(1,1))
        self.activation=nn.Sigmoid()
        self.pool2=nn.MaxPool2d((1,14*14),stride=(1,1))
    def forward(self,features):
        N,C,H,W=features.size()
        n_instances=H*W
        conv1=self.conv1(features)
        conv1=conv1.view(N,self.L,self.K,n_instances)
        pool1=self.pool1(conv1)
        act=self.activation(pool1)
        pool2=self.pool2(act)
        
        out=pool2.view(N,self.L)
        print('out',out[0])
        return out

