#import torch, torchvision
import mmdet 
#import numpy as np
import matplotlib.pyplot as plt
import mmcv
from datetime import datetime
import os

#from keras.model import load_model
#from keras.preprocessing import image

from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

base_dir = os.path.dirname(os.path.abspath('__file__'))

def load_model():
    config = base_dir + '/models/configs/default_config.py'
    checkpoint = base_dir + '/models/checkpoint/latest.pth'
    model = init_detector(config, checkpoint, device='cuda:0')
    return model 


def get_result_image(img):
    model = load_model()
    result = inference_detector(model, img)
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=0.3, show=False)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(mmcv.bgr2rgb(img))
    plt.tight_layout()

    now = datetime.now()
    dt = now.strftime("%m%d%H%M%S")

    result_img_path = './static/outputs/output_' + dt + '.png'
    plt.savefig(result_img_path)
    result_img_path = result_img_path[9:]
    return result_img_path
