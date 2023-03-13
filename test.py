import torch
import os
import argparse
import numpy as np
import cv2
import torch.nn.functional as F
from datasets.crowd import Crowd
# from models.fusion import fusion_model
from utils.evaluation import eval_game, eval_relative
# from models.CDINet.MyModel9 import CDINet
# from models.UCNet.ResNet_models import Generator
# from models.ModelTwo.One.TMC3Net import Net
# from ThreeM.DEFNet.DEFNet import fusion_model
# from ThreeM.A8.S9 import NetS
# from models.CAINet2.XR1 import MAINet
# from Three.Model1try.T_model import Net
# from models.fusion import fusion_model
from ThreeM.A8.lightstu.S9_3 import NetS
# from ThreeM.A8.XRsdfm import Net
# from models.MMNet.MMNet import FusionNet
from PIL import Image


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='/home/user/Crowd/bayes',
                        help='training data directory')
parser.add_argument('--save-dir', default='/media/user/shuju/Three/0218-090039kdstu',
                        help='model directory')
# parser.add_argument('--save-dir', default='/home/user/Crowd/BThree/0427-090510这个',
#                         help='model directory')
# parser.add_argument('--save-dir', default='/home/user/Crowd/BThree/0514-200356',
#                         help='model directory')
# parser.add_argument('--save-dir', default='/home/user/Crowd/bayes/0219-180949MMnet',
#                         help='model directory')
parser.add_argument('--model', default='best_model_27.pth'
                    , help='model name')

parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':

    datasets = Crowd(os.path.join(args.data_dir, 'test'), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = NetS()
    model.to(device)
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    # print(model.state_dict()['reg_layer2.0.weight'])
    # print(model.state_dict()['reg_layer3.0.bias'])
    # print(model.state_dict()['reg_layer3.2.weight'])
    # print(model.state_dict()['reg_layer3.2.bias'])
    # for name in model.state_dict():
    #     print(name)

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0

    for inputs, target, name in dataloader:
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)

        # inputs are images with different sizes
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs, _, _, _ = model(inputs)
            # print(outputs1)

            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error

        #     out = a1
        # # print(out.shape)
        #     path1 = os.path.join('/home/user/Crowd/Timg/111')
        #     out = F.interpolate(out, size=(320, 320), mode='bilinear', align_corners=False)
        #     out_img = out.cpu().detach().numpy()
        #     out_img = np.max(out_img, axis=1).reshape(320, 320)
        #     out_img = (((out_img - np.min(out_img))/(np.max(out_img) - np.min(out_img)))*255).astype(np.uint8)
        #     out_img = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
        #     cv2.imwrite(path1 + name[0] + '.png', out_img)
        #     print(path1 + name[0] + '.png')

        # # outputs = torch.sigmoid(-outputs)
        #     path = os.path.join('/home/user/Crowd/XRimg/QKV', name[0]+'.png')
        #     img = outputs.data.cpu().numpy()
        #     img = img.squeeze(0).squeeze(0)
        #     print(name[0], img.sum())
            # img = img * 255
            # img = Image.fromarray(np.uint8(img)).convert('L')
            # # img = img.resize((15, 20))
            # img.save(path)

            # path2 = os.path.join('/home/user/Crowd/Timg/222', name[0] + '.png')
            # img2 = a2.data.cpu().numpy()
            # img2 = img2.squeeze(0).squeeze(0)
            # print(name[0], img2.sum())
            # img2 = img2 * 255
            # img2 = Image.fromarray(np.uint8(img2)).convert('L')
            # # img = img.resize((15, 20))
            # img2.save(path2)

    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'MSE {mse:.2f} Re {relative:.4f}, '.\
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error)

    print(log_str)

