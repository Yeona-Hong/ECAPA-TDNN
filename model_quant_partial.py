'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as quant_nn

quant_desc_input = QuantDescriptor(num_bits=8, calib_method='histogram')
quant_desc_weight = QuantDescriptor(num_bits=8, calib_method='max', axis = 0)


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128, quantize=False):
        super(SEModule, self).__init__()
        self.quantize = quantize
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            quant_nn.QuantConv1d(channels, bottleneck, kernel_size=1, padding=0, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight) if quantize else nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            quant_nn.QuantConv1d(bottleneck, channels, kernel_size=1, padding=0, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight) if quantize else nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, quantize=False):
        super(Bottle2neck, self).__init__()
        self.quantize = quantize
        width = int(math.floor(planes / scale))
        self.conv1 = quant_nn.QuantConv1d(inplanes, width*scale, kernel_size=1, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight) if quantize else nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width*scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size/2) * dilation
        for i in range(self.nums):
            convs.append(quant_nn.QuantConv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight) if quantize else nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = quant_nn.QuantConv1d(width*scale, planes, kernel_size=1, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight) if quantize else nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, quantize=quantize)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 
class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()


        self.conv1  = quant_nn.QuantConv1d(80, C, kernel_size=5, stride=1, padding=2, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        # self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8, quantize = False)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8, quantize = False)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8, quantize = False)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        # self.layer4 = quant_nn.QuantConv1d(3*C, 1536, kernel_size=1, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            # quant_nn.QuantConv1d(4608, 256, kernel_size=1, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight),
            
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            # quant_nn.QuantConv1d(256, 1536, kernel_size=1, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        # self.fc6 = quant_nn.QuantLinear(3072, 192, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)
        self.fc6 = nn.Linear(3072, 192)

        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=False):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x