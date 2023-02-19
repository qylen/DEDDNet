import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
    
class Convdir(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(Convdir, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        
        self.relu = nn.ReLU(True)

    def forward(self, h, x):
        
        hx = torch.cat([h, x], dim=1)
        h = self.relu(self.convz(hx))
        
        return h
    
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
class BasicMotionEncoder(nn.Module):
    def __init__(self,arg):
        super(BasicMotionEncoder, self).__init__()
        # cor_planes = (1+2*arg.radius) * arg.num_level 
        cor_planes = (1+2*arg.radius) * 3
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 32, 7, padding=3)
        self.convf2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv = nn.Conv2d(16+64, 64-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
    
class BasicUpdateBlock(nn.Module):
    def __init__(self, arg,hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(arg)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=64+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 16*9, 1, padding=0))

    def forward(self, net,inp1, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp1, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        #delta_flow,mask = 0,0
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow