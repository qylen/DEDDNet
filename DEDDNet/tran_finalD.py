import argparse
import os
from unittest import TestLoader
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from utils import util, build_code_arch
from data.create_trainval_dataset import TrainDataset,ValDataset
# from models.kernel_yuan import KernelEDNet
from models.unet import UNet
#from models.MS_UNetD import KernelEDNet
from models.MS_UNetD_CVDE2 import KernelEDNet
from loss.deblur_loss import ReconstructLoss,ReconstructLossTest,PerceptualLoss,MonodepthLoss
from models.PatchGAN_D_DP import NLayerDiscriminator,DiscLossWGANGP,RelativisticDiscLossLS
# from models.models_6_server import KernelEDNet
import numpy as np
import time
###############monodepth##############
import networks
from torchvision.transforms import Resize

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='/home/zjc/dual_pixel/BaMBNet-main/option/train/Defocus_GAN_Trained.yaml', help='Defocus Deblur: Path to option ymal file.')
train_args = parser.parse_args()

opt, resume_state = build_code_arch.build_resume_state(train_args)
opt, logger, tb_logger = build_code_arch.build_logger(opt)

for phase, dataset_opt in opt['dataset'].items():
    if phase == 'train':
        train_dataset = TrainDataset(dataset_opt)
        train_loader = DataLoader(
            train_dataset, batch_size=dataset_opt['batch_size'], shuffle=True,
            num_workers=dataset_opt['workers'], pin_memory=True)
        logger.info('Number of train images: {:,d}'.format(len(train_dataset)))
assert train_loader is not None

Valdataset_opt = opt['dataset']['val']
test_dataset = ValDataset(Valdataset_opt)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        num_workers=dataset_opt['workers'], pin_memory=False)
logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_dataset)))



# create model
#model = UNet(3)
model = KernelEDNet()
optimizer = Adam(model.parameters(), betas=(opt['train']['beta1'], opt['train']['beta2']),
                 lr=opt['train']['lr'])

scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                     milestones=opt['train']['lr_steps'],
                                     gamma=opt['train']['lr_gamma'])
model = model.cuda()

# model_D=NLayerDiscriminator(input_nc=3,ndf=64,n_layers=3,use_sigmoid=False)
# model_D = model_D.cuda()
# optimizer_D = torch.optim.Adam(model_D.parameters(), betas=(opt['train']['beta1'], opt['train']['beta2']),
#                  lr=opt['train']['lr'] )

# scheduler_D = lr_scheduler.MultiStepLR(optimizer=optimizer_D,
#                                      milestones=opt['train']['lr_steps'],
#                                      gamma=opt['train']['lr_gamma'])
# resume training
if resume_state:
    logger.info('Resuming training from epoch: {}.'.format(
        resume_state['epoch']))
    start_epoch = resume_state['epoch']
    optimizer.load_state_dict(resume_state['optimizers'])
    scheduler.load_state_dict(resume_state['schedulers'])
    model.load_state_dict(resume_state['state_dict'])
    # if (resume_state['state_dict_D'] and resume_state['optimizers_D'] and resume_state['schedulers_D']) is not None:
    #     model_D.load_state_dict(resume_state['state_dict_D'])
    #     optimizer_D.load_state_dict(resume_state['optimizers_D'])
    #     scheduler_D.load_state_dict(resume_state['schedulers_D'])
else:
    start_epoch = 0

criterion = ReconstructLoss()
criterionT = ReconstructLossTest()
criter_P =PerceptualLoss()            # LG=0.5*Lp +0.006*Lx+0.01*Ladv 第一个是像素损失，第二个是感知损失，第三个是对抗loss
loss_GAN = DiscLossWGANGP()
loss_disparity = MonodepthLoss(
                n=1,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1)
#loss_GAN = RelativisticDiscLossLS()

# model = torch.nn.DataParallel(model)
# training
total_epochs = opt['train']['epoch']

max_steps = len(train_loader)
logger.info('Start training from epoch: {:d}'.format(start_epoch))
logger.info('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))
loss_epoch=[]
current_step = 0

###################MONODEPTH-PRETRAIN##############
device = torch.device("cuda")
model_path = opt['train']['monodepth_model_name']
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)

depth_decoder.to(device)
depth_decoder.eval()
for name, param in encoder.named_parameters():
    param.requires_grad = False
for name, param in depth_decoder.named_parameters():
    param.requires_grad = False



for epoch in range(start_epoch+1, total_epochs + 1):

    model.train()
    criterion.iter = epoch
    starttimeidx=time.time()
    for index, train_data in tqdm(enumerate(train_loader)):
        # training
        l_img, r_img, b_img, gt,cs_img,sample = train_data
        l_img = l_img.cuda()
        r_img = r_img.cuda()
        gt_img = gt.cuda()
        b_img = b_img.cuda()
        _,_,original_height, original_width = cs_img.shape
        torch_resize = Resize([feed_height,feed_width]) # 定义Resize类对象
        #usdipcs_img = torch_resize(cs_img).cuda()
        usdipcs_img = cs_img.cuda()
        features = encoder(usdipcs_img)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        # dispp1=disp_resized_np[0]
        # dispp2= disp_resized_np[1]
        cs_img = cs_img.cuda()
        #x = torch.cat((l_img,r_img,disp,b_img), dim=1)
        x = torch.cat((cs_img,disp,b_img), dim=1)
        #x = torch.cat((cs_img, b_img), dim=1)
        if epoch < 100:
            recover_img= model(x)
        else:
            recover_img =model(x,gt_img)
        #可以多一个warm up,先训练10代，然后再用GAN训练,然后生成器不用退火算法了，到GAN训练结束再微调
        # if epoch > 100:
        #     for iter_d in range(2):
        #         optimizer_D.zero_grad()
        #         loss_D = loss_GAN.get_loss(model_D,recover_img[0],gt_img,b_img)
        #         loss_D.backward()
        #         optimizer_D.step()
        #     #这个对抗loss也一样，10代后启用
        #     loss_gan=loss_GAN.get_g_loss(model_D,recover_img[0],gt_img,b_img)
        #     #这个对抗loss也一样，10代后启用
        #     loss_gan=loss_GAN.get_g_loss(model_D,cs_img,recover_img[0])
        losses = criterion(recover_img, gt_img)
        # 加感知损失,加GAN损失
        loss_P=criter_P.get_loss(recover_img[0],gt_img)
        
        #loss_disp = loss_disparity(disparity_map, [l_img,r_img])
        
        # if epoch >15:
        # #加L1正则化试一下
        #     regularization_loss = 0
        #     for param in model.parameters():
        #         regularization_loss += torch.sum(abs(param))            
        #     grad_loss = losses["total_loss"]+ 1e-5 * regularization_loss
        # else:
        # if epoch > 10:
        #     loss_pixel=losses["total_loss"]
        #     grad_loss = 0.7*loss_pixel+0.001*loss_P+0.01*loss_gan
        # else:
        # if epoch > 100:
        #     grad_loss = 0.5*losses["total_loss"]+0.006*loss_P+0.001*loss_gan
        # else:
        grad_loss = 0.5*losses["total_loss"]+0.006*loss_P#0.7 0.004
        # grad_loss = 0.5*losses["total_loss"]+0.006*loss_P #0.7 0.004
        # if epoch >= 10:
        #     grad_loss = 0.5*losses["total_loss"]+0.001*loss_gan
        # else:
        #     grad_loss = 0.5*losses["total_loss"]  #0.7 0.004

        optimizer.zero_grad()
        grad_loss.backward()
        optimizer.step()
        current_step = (epoch-1) * max_steps + index
        # log
        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
            epoch, current_step, scheduler.get_last_lr()[0])
        for k, v in losses.items():
            v = v.cpu().item()
            message += '{:s}: {:.4e} '.format(k, v)
            # tensorboard logger
            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                tb_logger.add_scalar(k, v, current_step)
        logger.info(message)
        loss_epoch.append(grad_loss.data.cpu())
    # update learning rate
    scheduler.step()
    endtimeidx =time.time()
    if epoch % 1 == 0:
        with util.Logger('./epoch_loss_val.txt'):
            print('# Test # epoch:{:4d} loss{:4f} train_time:{:4f}.'.format(epoch,float(np.array(loss_epoch).mean()),float((endtimeidx-starttimeidx)/60)) )
    loss_epoch=[]            
    
    # save models and training states
    if epoch % opt['logger']['save_checkpoint_freq'] == 0:
        logger.info('Saving models and training states.')
        save_filename = '{}_{}.pth'.format(epoch, 'models')
        save_path = os.path.join(opt['path']['models'], save_filename)
        state_dict = model.state_dict()
        save_checkpoint = {'state_dict': state_dict,
                           'optimizers': optimizer.state_dict(),
                           'schedulers': scheduler.state_dict(),
                           'epoch': epoch
                        #    'state_dict_D': model_D.state_dict(),
                        #    'optimizers_D': optimizer_D.state_dict(),
                        #    'schedulers_D': scheduler_D.state_dict()
                           }
        torch.save(save_checkpoint, save_path)
        torch.cuda.empty_cache()
    if epoch % opt['logger']['save_checkpoint_freq'] == 0:
        idx = 0
        tloss = 0
        model.eval()
        for test_data in tqdm(test_loader):
            with torch.no_grad():
                    l_img, r_img, b_img, gt,cs_img ,root_name = test_data
                    gt = gt.cuda()
                    l_img = l_img.cuda()
                    r_img = r_img.cuda()
                    b_img = b_img.cuda()
                    _,_,original_height, original_width = cs_img.shape
                    torch_resize = Resize([feed_height,feed_width]) # 定义Resize类对象
                    usdipcs_img = torch_resize(cs_img).cuda()
                    features = encoder(usdipcs_img)
                    outputs = depth_decoder(features)

                    disp = outputs[("disp", 0)]
                    disp_resized = torch.nn.functional.interpolate(
                        disp, (original_height, original_width), mode="bilinear", align_corners=False)
                    disp_resized_np = disp_resized.squeeze().cpu().numpy()
                    # dispp1=disp_resized_np[0]
                    # dispp2= disp_resized_np[1]
                    cs_img = cs_img.cuda()
                    x = torch.cat((l_img,r_img,disp_resized,b_img), dim=1)
                    # x = torch.cat((cs_img,disp_resized,b_img), dim=1)
                    # x = torch.cat((l_img,r_img, r_img),dim=1)
                    # x = torch.cat((cs_img, b_img), dim=1)
                    recover = model(x=x)[0][0]
                    tloss += util.calculate_psnr(gt, recover)
                    # losses=criterionT(recover,gt)
                    # loss=losses["total_loss"]
                    idx += 1
                    # tloss +=loss 
        tloss = tloss/idx
        torch.cuda.empty_cache()
        with util.Logger('./epoch_loss_val.txt'):
            print("val:epoch {} the valloss is {} ".format(epoch,tloss))
    
logger.info('Saving the final model.')
save_filename = 'latest.pth'
save_path = os.path.join(opt['path']['models'], save_filename)
save_checkpoint = {'state_dict': state_dict,
                           'optimizers': optimizer.state_dict(),
                           'schedulers': scheduler.state_dict(),
                           'epoch': epoch
                        #    'state_dict_D': model_D.state_dict(),
                        #    'optimizers_D': optimizer_D.state_dict(),
                        #    'schedulers_D': scheduler_D.state_dict()
                           }
torch.save(save_checkpoint, save_path)
logger.info('End of training.')
tb_logger.close()
