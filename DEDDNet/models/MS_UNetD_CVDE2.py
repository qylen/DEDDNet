# 多尺度去模糊
import torch.nn as nn
from models.rcab import RCAB, ResidualGroup, default_conv
import math
import torch
import torch.nn.functional as F
from models.unet_parts import *
from models.stereo_matching.extractor import BasicEncoder
from models.stereo_matching.corr import CorrBlock1D
from models.stereo_matching.util_s import coords_grid
from models.stereo_matching.update import BasicUpdateBlock,ConvGRU,Convdir

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def kernel_conv(kernel_size, input_dim, reduction, max_pool=False, upsample=False):
    res_conv = []
    out_dim = input_dim
    if max_pool ==True :
       out_dim = input_dim*2
    if upsample ==True :
       out_dim = input_dim/2
    
    if kernel_size <= 1:
        res_conv = [nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(True)]
        return nn.Sequential(*res_conv)  
    elif kernel_size ==2:
        res_conv = [
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            ]
        
    else:
        res_conv.append(ResidualGroup(default_conv, input_dim, 3, reduction, n_resblocks=math.floor(kernel_size/3)))

    if max_pool:
        res_conv.append(nn.MaxPool2d(kernel_size=2, stride=2))

    if upsample:
        res_conv.append(nn.Upsample(scale_factor=2))

    return nn.Sequential(*res_conv)


def connect_conv(input_dim, output_dim, kernel_size, stride, padding, bias=True, dilation=1):
    conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, dilation=dilation)
    relu = nn.ReLU(True)
    #ln = nn.InstanceNorm2d(output_dim)

    return nn.Sequential(*[conv, relu])

class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.ReLU(True))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V       
class CasualUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True,scale_factor=2,bias=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.conv = nn.Conv2d(in_channels,out_channels,1,stride=1,padding=0,bias=bias)
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
    def forward(self,x):
        return self.up(self.conv(x))
            

class KernelEDNet(nn.Module):
    def __init__(self):
        super(KernelEDNet, self).__init__()
        kernel_size = [7,10] #本来是[1,4,7,10]
        self.kernel_size= kernel_size
        self.channel = 64
        self.head = connect_conv(3, self.channel, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        self.headdepth = nn.Sequential(connect_conv(1, self.channel//4, kernel_size=3, stride=1, padding=1, bias=True, dilation=1),
                                       kernel_conv(4,self.channel//4,16))
        convk_tail = nn.Conv2d(self.channel * 1, 3, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        reluk_tail = nn.Sigmoid()
        self.tail_hard = nn.Sequential(*[convk_tail, reluk_tail])
        
        # self.connect = nn.Conv2d(self.channel * 1, self.channel, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        # self.connect2 = nn.Conv2d(self.channel * 1, self.channel, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        
        # self.unetdown4 = Down4(self.channel,self.channel)        
        # self.unetdown2 = Down(self.channel,self.channel)
        # self.unetup2 = Up(self.channel,self.channel,True)
        # self.unetup4 = Up4(self.channel,self.channel)
        
        # self.l_r_m = nn.Conv2d(self.channel * 2, self.channel, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        
        # self.skff = SKFF(int(self.channel), 2)
        # self.cup2 = CasualUp(self.channel,self.channel,scale_factor=2)
        # self.cup4 = CasualUp(self.channel,self.channel,scale_factor=4)
        # self.cup8 = CasualUp(self.channel,self.channel,scale_factor=8)
        # self.cup16 = CasualUp(self.channel,self.channel,scale_factor=16)


        self.layer1 = nn.ModuleList()
        for k in kernel_size:
            self.layer1.append(kernel_conv(k, self.channel, 16, max_pool=True)) #256
 
        self.layer2 = nn.ModuleList()
        for k in kernel_size:
            self.layer2.append(kernel_conv(k, self.channel, 16, max_pool=True))  #128
 
        self.layer3 = nn.ModuleList()
        for k in kernel_size:
            self.layer3.append(kernel_conv(k,self.channel, 16, max_pool=True))  #64 

        self.layer4 = nn.ModuleList()
        for k in kernel_size:
            self.layer4.append(kernel_conv(k, self.channel, 16, max_pool=True))  #32

        self.layer5 = nn.ModuleList()
        for k in kernel_size:
            self.layer5.append(kernel_conv(k, self.channel, 16, upsample=True))  #64

        self.layer6 = nn.ModuleList()
        for k in kernel_size:
            self.layer6.append(kernel_conv(k, self.channel, 16, upsample=True))   #128

        self.layer7 = nn.ModuleList()
        for k in kernel_size:
            self.layer7.append(kernel_conv(k, self.channel, 16, upsample=True))   #256

        self.layer8 = nn.ModuleList()
        for k in kernel_size:
            self.layer8.append(kernel_conv(k, self.channel, 16, upsample=True))    #512
            
        
        self.layerGRU64 = nn.ModuleList()
        self.layerGRU128 = nn.ModuleList()
        self.layerGRU256 = nn.ModuleList()
        self.layerGRU512 = nn.ModuleList()
        for k in kernel_size:
            self.layerGRU64.append(ConvGRU(hidden_dim=64,input_dim=16))    #64
        for k in kernel_size:
            self.layerGRU128.append(ConvGRU(hidden_dim=64,input_dim=16))    #128
        for k in kernel_size:
            self.layerGRU256.append(ConvGRU(hidden_dim=64,input_dim=16))    #256
        for k in kernel_size:
            self.layerGRU512.append(ConvGRU(hidden_dim=64,input_dim=16))    #512
        
        ##############直接融合
        # for k in kernel_size:
        #     self.layerGRU64.append(Convdir(hidden_dim=64,input_dim=16))    #64
        # for k in kernel_size:
        #     self.layerGRU128.append(Convdir(hidden_dim=64,input_dim=16))    #128
        # for k in kernel_size:
        #     self.layerGRU256.append(Convdir(hidden_dim=64,input_dim=16))    #256
        # for k in kernel_size:
        #     self.layerGRU512.append(Convdir(hidden_dim=64,input_dim=16))    #512
            
        ###########输入融合直接在开头多加一通道
        
        
            
        
        
        



        self.MAX_TRAINNUM = 2
        self.iter_num = 10
    
    
    
    def forward(self, x, gt=None):
        blur, _ = x[:, 4:, :, :].abs().max(dim=1, keepdim=True)
        #monodepth = x[:,3:6,:,:]
        x = x[:, :4, :, :]
        # ############################stereo_matching
        
        
        
        
        
        xmap1 = self.head(x[:,:3,:,:])
        depthprymaid=[x[:,-1:,:,:]]
        depthtrait=[]
        for i in range(1,4):
            temp=F.interpolate(depthprymaid[i-1], scale_factor=0.5, mode='bilinear')
            depthprymaid.append(temp)
        for i in range(4):
            depthtrai = self.headdepth(depthprymaid[i])
            depthtrait.append(depthtrai)
        
        # xmap1 = self.head(x_l)
        # xmap2 = self.head(x_r)
        
        #stereo = []
                   
        ############################
        
        
        blur_mask = []
        # blur_mask_d1=[]
        # blur_mask_d2=[]
        
        # blur_d1 = F.interpolate(blur,scale_factor=0.25)
        # blur_d2 = F.interpolate(blur,scale_factor=0.125)

        feature_layer = []
        # feature_layer_d4 = []
        # feature_layer_d8 = []

        if gt is not None:
            self.iter_num += 1

        static_kernel_size =  [0.0,1.9]#[0.0,1.8,4.5,6.9]
        for kernel_bound, kernel_up in zip(static_kernel_size, static_kernel_size[1:]):
            mask = ((blur >= kernel_bound) & (blur < kernel_up)).float()
            blur_mask.append(mask)

        mask = (blur >= static_kernel_size[-1]).float()
        blur_mask.append(mask)
        

        ######################
        # #x为原分辨率
        res_x = 0
        layer_output1,layer_output1r = [],[]
        for i in range(len(self.kernel_size)):
            layer_output1.append(self.layer1[i](xmap1))                    #256     
        
        

        layer_output2,layer_output2r = [],[]
        for i in range(len(self.kernel_size)):
            #res_x = F.adaptive_avg_pool2d(xmap1, layer_output1[i].size()[2:])
            layer_output2.append(self.layer2[i](res_x+layer_output1[i]))           #128  
        
        
        

        layer_output3,layer_output3r = [],[]
        for i in range(len(self.kernel_size)):
            #res_x = F.adaptive_avg_pool2d(xmap1, layer_output2[i].size()[2:])
            layer_output3.append(self.layer3[i](res_x+layer_output2[i]))             #64       
        
        
        layer_output4,layer_output4r = [],[]
        for i in range(len(self.kernel_size)):
            #res_x = F.adaptive_avg_pool2d(xmap1, layer_output3[i].size()[2:])             #32
            layer_output4.append(self.layer4[i](res_x+layer_output3[i]))
        

        layer_output5,layer_output5r = [],[]
        for i in range(len(self.kernel_size)):
            #res_x = F.adaptive_avg_pool2d(xmap1, layer_output4[i].size()[2:])
            layer_output5.append(self.layer5[i](res_x + layer_output4[i]))           #64
            if i ==-1:
                ik =0
            else:
                ik=-1
            layer_output5[i]=self.layerGRU64[i](layer_output5[i],depthtrait[ik])
        
        
        
        
        
    


        layer_output6,layer_output6r = [],[]
        for i in range(len(self.kernel_size)):
            #res_x = F.adaptive_avg_pool2d(xmap1, layer_output3[i].size()[2:])
            layer_output6.append((self.layer6[i](res_x+layer_output5[i]+ layer_output3[i])))     #128
            if i ==-1:
                ik =0
            else:
                ik=-2
            layer_output6[i]=self.layerGRU128[i](layer_output6[i],depthtrait[ik])
        

        

        layer_output7,layer_output7r = [],[]
        for i in range(len(self.kernel_size)):
            #res_x = F.adaptive_avg_pool2d(xmap1, layer_output6[i].size()[2:])
            layer_output7.append((self.layer7[i](res_x+layer_output6[i] + layer_output2[i])))     #256
            if i ==-1:
                ik =0
            else:
                ik=-3
            layer_output7[i]=self.layerGRU256[i](layer_output7[i],depthtrait[ik])
        


        layer_outputx = []
        for i in range(len(self.kernel_size)):
            #res_x = F.adaptive_avg_pool2d(xmap1, layer_output7[i].size()[2:])
            layer_outputx.append(self.layer8[i](res_x+layer_output7[i] + layer_output1[i]))       #512
            layer_outputx[i]=self.layerGRU512[i](layer_outputx[i],depthtrait[0])
       
        
        if self.iter_num < self.MAX_TRAINNUM:
            iter_weight = torch.exp(torch.tensor(- (self.iter_num * 2 / self.MAX_TRAINNUM) ** 2))
            for layer_i, blur_i in zip(layer_outputx, blur_mask):
                feature_layer.append((layer_i * blur_i * iter_weight + (1-iter_weight) * layer_i).unsqueeze(0))
            
        else:
            feature_layer = [layer_i.unsqueeze(0) for layer_i in layer_outputx]
        out = self.tail_hard(xmap1+ torch.cat(feature_layer, dim=0).sum(dim=0))
        return [out]
