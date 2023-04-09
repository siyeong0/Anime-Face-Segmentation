import torch
from torchvision import transforms
import cv2 as cv
from PIL import Image
import argparse

from network import UNet
from util import seg2img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', help='path of source image(square)')
    parser.add_argument('--save_dir', help='directory to save segmentation image')
    parser.add_argument('--model_path', default='model/UNet.pth', help='path to load trained U-Net model')
    args = parser.parse_args()
    
    src_path = args.src_path
    save_dir = args.save_dir
    model_path = args.model_path
    
    model = UNet()
    model.load_state_dict(torch.load(model_path))

    transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),])

    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        model.eval()
        
        img = Image.open(src_path)
        img = transform(img).unsqueeze(dim=0).cuda()
        
        seg = model(img).squeeze(dim=0)
        
        result = seg2img(seg.cpu().detach().numpy())
        cv.imwrite(save_dir+'result.png', result)
    
    

