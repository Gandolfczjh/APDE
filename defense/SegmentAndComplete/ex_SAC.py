import numpy as np
import cv2
import torch
import os

import argparse
from tqdm import tqdm
from patch_detector import PatchDetector
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

device = 'cuda'

ori_path = '/data/zjh/code/defense/SegmentAndComplete/ckpts/SAC_original.pth'
retrain_path = '/data/zjh/code/defense/SegmentAndComplete/ckpts/SAC_retrained.pth'

SAC_processor = PatchDetector(3, 1, base_filter=16, device=device, square_sizes=[150, 100, 75, 50, 25], n_patch=1)
SAC_processor.unet.load_state_dict(torch.load(retrain_path, map_location=device))

def output(advdataset_path):
    advdataset_path = '/data/zjh/stop_sign/'
    advimg_list = os.listdir(advdataset_path)

    # attack_detector = advdataset_path.split('/')[-2] + '/' + advdataset_path.split('/')[-1]
    # attack_detector = ''
    # mask_path = '/data/zjh/dataset/adv_dataset/AA_Mask/' + attack_detector + '/SAC/'
    # masked_path = '/data/zjh/dataset/adv_dataset/AA_Masked/' + attack_detector + '/SAC/'

    # attack_detector = advdataset_path.split('/')[-2] + '/' + advdataset_path.split('/')[-1]
    # masked_path = '/data/zjh/dataset/adv_dataset/Masked2/' + attack_detector + '/SAC_retrain/'
    # mask_path = '/data/zjh/dataset/adv_dataset/Mask2/' + attack_detector + '/SAC_retrain/'

    masked_path = '/data/zjh/stop_sign_masked/'
    # mask_path = '/data/zjh/jh_SAC/benign_mask_path/'

    # os.makedirs(mask_path, exist_ok=True)
    os.makedirs(masked_path, exist_ok=True)

    for i in tqdm(range(len(advimg_list))):

        img = cv2.imread(os.path.join(advdataset_path, advimg_list[i]))
        image_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        image = np.stack([image_0], axis=0).astype(np.float32)/255.0 

        image=image.transpose(0,3,1,2)
        image1=torch.tensor(image).to(device).float()
        x_processed, mask_list, _ = SAC_processor(image1, bpda=True, shape_completion=False)
        image_sac = np.asarray(x_processed[0].cpu().detach()*255, dtype=np.uint8)
        image_sac =image_sac.transpose(1,2,0)
        image_sac = cv2.cvtColor(image_sac, cv2.COLOR_RGB2BGR)

        mask_tensor = torch.cat([mask_list[0], mask_list[0], mask_list[0]], dim=1)[0].permute(1,2,0).cpu().detach() * 255
        mask_numpy = np.array(mask_tensor.numpy(), dtype=np.uint8)

        cv2.imwrite(masked_path + advimg_list[i], image_sac)
        # cv2.imwrite(mask_path + advimg_list[i], mask_numpy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--advdataset_path', type=str, default='/data/zjh/dataset/mc/SAC')
    args = parser.parse_args()
    output(args.advdataset_path)