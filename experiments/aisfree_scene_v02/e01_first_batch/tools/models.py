"""Minimal E01 architectures, torch only (no torchvision or weight download).
This is a ResNet-18-style SAR stem + GroupNorm BASELINE, not a novel backbone.
"""
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,a,b,stride=1):
        super().__init__()
        self.f=nn.Sequential(nn.Conv2d(a,b,3,stride,1,bias=False),nn.GroupNorm(32,b),nn.ReLU(inplace=True),
                             nn.Conv2d(b,b,3,1,1,bias=False),nn.GroupNorm(32,b))
        self.skip=nn.Identity() if (a==b and stride==1) else nn.Sequential(nn.Conv2d(a,b,1,stride,bias=False),nn.GroupNorm(32,b))
        self.act=nn.ReLU(inplace=True)
    def forward(self,x):return self.act(self.f(x)+self.skip(x))

class SARResNet18GN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(2,64,3,1,1,bias=False),nn.GroupNorm(32,64),nn.ReLU(inplace=True))
        self.layers=nn.Sequential(Block(64,64),Block(64,64),Block(64,128,2),Block(128,128),
                                  Block(128,256,2),Block(256,256),Block(256,512,2),Block(512,512))
        self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(512,num_classes)
    def encode(self,x):return self.pool(self.layers(self.stem(x))).flatten(1)
    def forward(self,x):return self.fc(self.encode(x))

class CandidateHead(nn.Module):
    def __init__(self,in_dim,classes,architecture):
        super().__init__()
        if architecture=='linear':self.net=nn.Linear(in_dim,classes)
        elif architecture=='mlp128':self.net=nn.Sequential(nn.Linear(in_dim,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,classes))
        else:raise ValueError('Unknown architecture')
    def forward(self,x):return self.net(x)

class RelationMomentModel(nn.Module):
    def __init__(self,in_dim,classes,D=4):
        super().__init__()
        self.classes=classes
        self.net=nn.Sequential(nn.Linear(in_dim+classes,64),nn.SiLU(),nn.Linear(64,D))
    def logits(self,x,class_ids):
        onehot=torch.nn.functional.one_hot(class_ids,num_classes=self.classes).to(x.dtype)
        return self.net(torch.cat([x,onehot],dim=-1))
    def forward(self,x,class_ids):return self.logits(x,class_ids).sigmoid()
