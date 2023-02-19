import argparse
import os
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import util

from data.create_test_dataset_other import TestDataset



from models.MS_UNetD_CVDE2 import KernelEDNet



from loss.deblur_loss import ReconstructLossTest
import option.options as option
import cv2

from thop import profile
from torchstat import stat

import networks
from torchvision.transforms import Resize

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='/home/zjc/dual_pixel/DEDDNet/option/test/Deblur_Dataset_Test.yaml', help='Defocus Deblur: Path to option ymal file.')
test_args = parser.parse_args()

opt = option.parse(test_args.opt, is_train=False)
util.mkdir_and_rename(opt['path']['results_root'])  # rename results folder if exists
util.mkdirs((path for key, path in opt['path'].items() if not key == 'results_root'
                     and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)

logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

torch.backends.cudnn.deterministic = True
# convert to NoneDict, which returns None for missing keys
opt = option.dict_to_nonedict(opt)


dataset_opt = opt['dataset']['test']
test_dataset = TestDataset(dataset_opt)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        num_workers=dataset_opt['workers'], pin_memory=True)
logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_dataset)))

#model = UNet(3)
model = KernelEDNet()
# input = torch.randn( 1,5, 512, 512)
# macs, params = profile(model, inputs=(input, ))
# print("macs:{}G,params:{}M".format(macs/(10**9),params/(10**6)))
# stat(model,(6,1120,1680))
#model=torch.nn.DataParallel(model).cuda()
# resume for test
device_id = torch.cuda.current_device()
resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))

start_epoch = resume_state['epoch']

logger.info('Resuming training from epoch: {}.'.format(
    resume_state['epoch']))
model.load_state_dict(resume_state['state_dict'],strict=False)
model.iter_num = 100e4

model = model.cuda()

###################MONODEPTH-PRETRAIN##############
device = torch.device("cuda")
model_path = opt['monodepth_model_name']
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

criterionT = ReconstructLossTest()
# testing
max_steps = len(test_loader)

torch.cuda.empty_cache()
avg_psnr = 0.0
avg_ssim = 0.0
avg_mae = 0.0
avg_lpips = 0.0
idx = 0
model.eval()
for test_data in tqdm(test_loader):
    with torch.no_grad():
        gt,cs_img, root_name,_ = test_data
        #l_img, r_img, gt,cs_img, root_name = test_data
        
        # l_img = l_img.cuda()
        # r_img = r_img.cuda()
        gt = gt.cuda()
        
        _,_,original_height, original_width = cs_img.shape
        torch_resize = Resize([feed_height,feed_width]) # 定义Resize类对象
        usdipcs_img = torch_resize(cs_img).cuda()
        features = encoder(usdipcs_img)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        # disp_resized_np = disp_resized.squeeze().cpu().numpy()
        
        
        cs_img=cs_img.cuda()
        x = torch.cat((cs_img,disp_resized,cs_img), dim=1)
        #x = torch.cat((cs_img, cs_img,cs_img), dim=1)
        recover = model(x)[0][0]
        #disp = (model(x)[1][0]).squeeze().cpu().numpy()
        
        # Save ground truth
        img_dir = opt['path']['test_images']
        recover_img = (recover.squeeze().cpu() *65535.0).permute(1, 2, 0)
        recover_img = recover_img.clamp(0,65535)
        recover_img = recover_img.numpy().astype(np.uint16)
        save_img_path_gtr = os.path.join(img_dir,
                                         "{:s}_recover.png".format(root_name[0][0]))
        cv2.imwrite(save_img_path_gtr, recover_img)
        # calculate psnr
        idx += 1
        avg_psnr += util.calculate_psnr(gt, recover)
        logger.info("current {} psnr is {:.4e}".format(root_name[0][0] ,util.calculate_psnr(gt, recover)))
        avg_ssim += util.calculate_ssim(gt, recover)
        avg_mae += util.calculate_mae(gt, recover)
        avg_lpips += util.calculate_lpips(gt.cpu(), recover.cpu())
        torch.cuda.empty_cache()

avg_psnr = avg_psnr / idx
avg_ssim = avg_ssim / idx
avg_mae = avg_mae / idx
avg_lpips = avg_lpips / idx
# log
logger.info('# Test # psnr: {:.4e} ssim: {:e} mae: {:4e} lpips: {:4e}.'.format(avg_psnr, avg_ssim, avg_mae, avg_lpips))
logger_test = logging.getLogger('test')  # validation logger
with util.Logger('./epoch_loss_val.txt'):
    print('Test: epoch {} psnr: {:.4e} ssim: {:e} mae: {:4e} lpips: {:4e}.'.format(start_epoch,avg_psnr, avg_ssim, avg_mae, avg_lpips))
logger_test.info('Test psnr: {:.4e} ssim: {:e} mae: {:4e} lpips: {:4e}.'.format(avg_psnr, avg_ssim, avg_mae, avg_lpips))
logger.info('End of testing.')

# idx = 0
# tloss = 0
# model.eval()
# for test_data in tqdm(test_loader):
#     with torch.no_grad():
#             l_img, r_img,  gt,cs_img ,root_name = test_data
#             gt = gt.cuda()
#             l_img = l_img.cuda()
#             r_img = r_img.cuda()
#             #b_img = b_img.cuda()
#             cs_img=cs_img.cuda()
#             #x = torch.cat((l_img,r_img, r_img),dim=1)
#             x = torch.cat((cs_img, cs_img), dim=1)
#             recover = model(x=x)[0]
#             losses=criterionT(recover,gt)
#             loss=losses["total_loss"]
#             idx += 1
#             tloss +=loss 
# tloss = tloss/idx
# torch.cuda.empty_cache()
# with util.Logger('./epoch_loss_val.txt'):
#     print("val:epoch {} the testloss is {} ".format(start_epoch,tloss.data.cpu()))
