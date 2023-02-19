from models.unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)  ##512
        ##########encoder###################
        self.down1 = Down(64, 128)        #256
        self.down2 = Down(128, 128)          #128
        self.down3 = Down(128, 128)        #64
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 128 // factor)  #32
        ##########encoder###################
        
        ###########decoder###################
        

        ##########for left_right_DP##############
        self.up1 = Up(128, 128 // factor, bilinear)   #64
        self.up2 = Up(128, 128 // factor, bilinear)     #128
        self.up3 = Up(128, 64 // factor, bilinear)     #256
        self.up4 = Up(64, 64, bilinear)          #512
        self.out = OutConv(64, 3)


    def forward(self, x):
        blur, _ = x[:, 4:, :, :].abs().max(dim=1, keepdim=True)
        #monodepth = x[:,3:6,:,:]
        x = x[:, :3, :, :]
        #######encoder##############
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        ########decoder#############

        x6=self.up1(x5,x4)
        x7=self.up2(x6,x3)
        x8=self.up3(x7,x2)
        x9=self.up4(x8,x1)
        xout=self.out(x9)

        return [xout]


