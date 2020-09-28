from model import resnet
import data as dataset
import numpy as np
import cv2
import argparse
import os
import torch
import torch.backends.cudnn as cudnn

import argparse

parser = argparse.ArgumentParser(description='Grad-Cam-Pytroch')
parser.add_argument('--pretrained', default='./confidence_128/model.pth', type=str, help='pretrained model directory')
parser.add_argument('--gpu_num', default='0', type=str, help='use gpu number')
parser.add_argument('--input_size', default=32, type=int, help='input size')
args = parser.parse_args()

# set GPU ID
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
cudnn.benchmark = True
# check save path
if not os.path.exists('./cam/'):
    os.makedirs('./cam/')

if not os.path.exists('./cam_orign'):
    os.makedirs('./cam_orign')

# make dataloader
train_loader, test_loader = dataset.get_loader()

# set model
model = resnet.ResNet34().cuda()
pre_trained_net = args.pretrained
model.load_state_dict(torch.load(pre_trained_net))

mean = [0.5, 0.5, 0.5]
stdv = [0.5, 0.5, 0.5]

def forward_hook(module, input, output):
    hook_outputs.append(torch.squeeze(output))
def backward_hook(module, input, output):
    hook_outputs.append(torch.squeeze(output[0]))

# cam_layer = model.module.layer4[2].conv3
# cam_layer = model.layer4[1].conv2
cam_layer = model.layer4[2].conv2
cam_layer.register_forward_hook(forward_hook)
cam_layer.register_backward_hook(backward_hook)

for i, (input, target) in enumerate(test_loader):
    if i == 500:
        break

    hook_outputs = []
    input = input.cuda()
    target = target.long().cuda()
    output = model(input).squeeze()
    output[target].backward(retain_graph=True)
    a_k = torch.mean(hook_outputs[1], dim=(1,2), keepdim=True)
    cam_out = torch.sum(a_k * hook_outputs[0], dim=0)
    # normalise
    cam_out = (cam_out+torch.abs(cam_out))/2
    cam_out = cam_out/torch.max(cam_out)
    # bilinear upsampling     scale_factor = 64(intput image szie)
    upsampling = torch.nn.Upsample(scale_factor = args.input_size/len(cam_out), mode='bilinear', align_corners=False)
    resized_cam = upsampling(cam_out.unsqueeze(0).unsqueeze(0)).detach().squeeze().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)
    original_img = input.squeeze(dim=0)

    for j in range(len(mean)):
        original_img[j] *= stdv[j]
        original_img[j] += mean[j]

    original_img = np.array(original_img.permute(1, 2, 0).cpu() * 255.0)
    cam_img = heatmap * 0.3 + original_img * 0.5
    cv2.imwrite('./cam/grad_cam_{0}_{1}.png'.format(i, target.cpu().numpy()), cam_img)
    cv2.imwrite('./cam_orign/origin_{0}_{1}.png'.format(i,target.cpu().numpy()), original_img*1)
#
