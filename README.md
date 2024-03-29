# DEDDNet
the code of paper of EFFICIENT FUSION OF DEPTH INFORMATION FOR DEFOCUS DEBLURRING-ICASSP2024

The link to the pre-trained model of DEDDNet is:
https://drive.google.com/file/d/1DnVfActIucVEVyZw0rX3Mdx7-K4mYRvt/view?usp=sharing

The link to the pre-trained model of MonoDepthV2 is:
https://drive.google.com/file/d/1DH8VL_dv2dkMbcHk7v6dMlNux0ggJsOn/view?usp=sharing

After downloading the file, put the models in
experiment\

monodepthmodels\

for testing, modify \option\test\Deblur_Dataset_Test.yaml for your dataset. Run test_D.py.

for training, modify \option\test\Defocus_GAN_Trained.yaml for your dataset.
And then you can use train_finalD.py for training


