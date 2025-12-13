# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
from torchvision import transforms
from torchvision.utils import save_image
import cv2
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class my_inria(Dataset):
    def __init__(self, images_path, input_size, is_augment=False, return_img_name=False):
        self.images_path = images_path
        self.imgs = os.listdir(images_path)
        self.input_size = input_size
        self.n_samples = len(self.imgs)
        # is_augment = False
        self.transform = transforms.Compose([])
        if is_augment:
            self.transform = self.transform_fn
        self.ToTensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        self.return_img_name = return_img_name

    def transform_fn(self, im, p_aug=0.5):
        """This is for random preprocesser augmentation of p_aug probability

        :param im:
        :param p_aug: probability to augment preprocesser.
        :return:
        """
        gate = torch.tensor([0]).bernoulli_(p_aug)
        if gate.item() == 0: return im
        im_t = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # transforms.RandomResizedCrop((416, 416), scale=(0.2, 0.9)),
            transforms.RandomRotation(5),
        ])(im)

        return im_t

    def pad_scale(self, img):
        """Padding the img to a square-shape to avoid stretch from the Resize op.

        :param img:
        :return:
        """
        w, h = img.size
        if w == h:
            return img

        pad_size = int((w - h) / 2)
        if pad_size < 0:
            pad = (abs(pad_size), 0)
            side_len = h
        else:
            side_len = w
            pad = (0, pad_size)

        padded_img = Image.new('RGB', (side_len, side_len), color=(127, 127, 127))
        padded_img.paste(img, pad)
        return padded_img

    def __getitem__(self, index):
        # print(self.imgs[index], index)
        img_path = os.path.join(self.images_path, self.imgs[index])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        # image = self.pad_scale(image)

        if self.return_img_name:
            return self.ToTensor(image), self.imgs[index]

        return self.ToTensor(image)

    def __len__(self):
        return self.n_samples


def create_image_from_boxes(boxes, image_size=(416, 416)):
    # 创建全黑的图像
    image = np.zeros(image_size, dtype=np.uint8)

    # 遍历每个检测框
    for box in boxes:
        x1, y1, x2, y2 = box
        # 在检测框的区域内设置像素为255
        image[y1:y2, x1:x2] = 255

    return image


def feature_shield(imgs, sigma=3.0, threshold_factor=2.0):
    '''
    Input:
        imgs (batch size, channel, height, width)
    '''
    NF_smooth = transforms.GaussianBlur(3, sigma)
    nb, _, height, width = imgs.shape  # batch size, channels, height, width

    # high-pass / low-pass filter
    lpf = torch.zeros((height,width))
    R = (height+width)//8
    for x in range(width):
        for y in range(height):
            if ((x-(width-1)/2)**2 + (y-(height-1)/2)**2) < (R**2):
                lpf[y,x] = 1
    hpf = 1-lpf
    hpf, lpf = hpf.to(imgs.device), lpf.to(imgs.device)
    

    im_copy=imgs.clone()
    mask = torch.zeros_like(imgs)
    mask_background = torch.ones_like(imgs)
    f = fft.fftn(im_copy,dim=(2,3))
    f = torch.roll(f,(height//2,width//2),dims=(2,3)) 
    f_l = f * lpf
    f_l = torch.roll(f_l,(-height//2,-width//2),dims=(2,3))
    X_l = torch.abs(fft.ifftn(f_l,dim=(2,3)))
    X_l = torch.clamp(X_l,0,1)
    X_l = torch.mean(X_l, dim=1)

    # Region selection
    sigma, std = torch.mean(X_l, dim=(1,2)), torch.std(X_l, dim=(1,2))
    for idx in range(X_l.shape[0]):
        X_l[idx] = torch.where(abs(X_l[idx]-sigma[idx]) > threshold_factor * std[idx], 1, 0)
    mask = X_l.unsqueeze(1).repeat(1,3,1,1)
    mask_background -= mask

    # suppress
    im_copy = NF_smooth(im_copy).clamp_(0,1)
    imgs = im_copy * mask + imgs * mask_background

    return imgs



def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='train',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        NFSI=0,
        thres_factor=3.0,
        sigma=3.0,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        advdataset_path=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        # data = check_dataset(data)  # check
    

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    # is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    is_coco = 1
    # nc = 1 if single_cls else int(data['nc'])  # number of classes
    nc = 1
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()


    # Dataloader
    if not training:
        # if pt and not single_cls:  # check --weights are trained on --data
        #     ncm = model.model.nc
        #     assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
        #                       f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        
        # data_path = '/data/zjh/dataset/yolov3/'
        # dataloader = create_dataloader(data[task],
        #                                imgsz,
        #                                batch_size,
        #                                stride,
        #                                single_cls,
        #                                pad=pad,
        #                                rect=rect,
        #                                workers=workers,
        #                                prefix=colorstr(f'{task}: '))[0]
        # advdataset_path = '/data/zjh/dataset/adv_dataset/images/advpatch/centernet'
        # advdataset_path = '/data/zjh/dataset/adv_dataset/out/tshirt/yolov2'
        train_dataset = my_inria(advdataset_path, 448, is_augment=False, return_img_name=True)
        dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

        # attack_detector = advdataset_path.split('/')[-2] + '/' + advdataset_path.split('/')[-1]
        # attack_detector = ''
        # mask_path = '/data/zjh/dataset/adv_dataset/AA_Mask/'+ attack_detector + '/NAPGuard/'
        # masked_path = '/data/zjh/dataset/adv_dataset/AA_Masked/'+ attack_detector + '/NAPGuard/'

        # attack_detector = advdataset_path.split('/')[-2] + '/' + advdataset_path.split('/')[-1]
        # mask_path = '/data/zjh/dataset/size_exp/Mask/' + advdataset_path.split('/')[-3] + '/' + attack_detector + '/NAPGuard/'
        # masked_path = '/data/zjh/dataset/size_exp/Masked/'+ advdataset_path.split('/')[-3] + '/'+ attack_detector + '/NAPGuard_retrain/'

        # attack_detector = advdataset_path.split('/')[-2] + '/' + advdataset_path.split('/')[-1]
        # # mask_path = '/data/zjh/dataset/adv_dataset/out_mask/' + attack_detector + '/NAPGuard/'
        # masked_path = '/data/zjh/dataset/adv_dataset/out_masked/' + attack_detector + '/NAPGuard/'

        masked_path = '/data/zjh/code/eval/data/xm_eval/adv_imgs/retrain_NAPGuard/T-SEA/'
        # mask_path = '/data/zjh/jh_NAP/benign_mask_path/'

        # os.makedirs(mask_path, exist_ok=True)
        os.makedirs(masked_path, exist_ok=True)


    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    start_time=time.time()

    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, im_path) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                # targets = targets.to(device)
            # print(targets)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            # im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
        
        # Inference
        with dt[1]:
            # NFSI Strategy
            if NFSI:
                im = feature_shield(im, sigma=sigma, threshold_factor=thres_factor)
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # # NMS
        # targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # # print(targets)
        # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        0.45,
                                        iou_thres,
                                        labels=(),
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)
            # print(preds)
        
        for im_i in range(nb):
            bbox = np.array(preds[im_i][:, :4].cpu().numpy()/448*416, dtype=np.int64)
            mask0 = create_image_from_boxes(bbox)
            mask = np.tile(np.expand_dims(mask0, axis=-1), (1,1,3))

            img = cv2.imread(os.path.join(advdataset_path, im_path[im_i]))
            img1 = torch.where(torch.from_numpy(mask) > 100, 0, torch.from_numpy(img))

            cv2.imwrite(os.path.join(masked_path, im_path[im_i]), np.array(img1.numpy(), dtype=np.uint8))
            # cv2.imwrite(os.path.join(mask_path, im_path[im_i]), mask)

    #     # Metrics
    #     for si, pred in enumerate(preds):
    #         labels = targets[targets[:, 0] == si, 1:]

    #         nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
    #         path, shape = Path(paths[si]), shapes[si][0]

    #         correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
    #         seen += 1

    #         if npr == 0:
    #             if nl:
    #                 stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
    #                 if plots:
    #                     confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
    #             continue

    #         # Predictions
    #         if single_cls:
    #             pred[:, 5] = 0
    #         predn = pred.clone()
    #         scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

    #         # Evaluate
    #         if nl:
    #             tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
    #             scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
    #             labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
    #             correct = process_batch(predn, labelsn, iouv)
    #             if plots:
    #                 confusion_matrix.process_batch(predn, labelsn)
    #         stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    #         # Save/log
    #         if save_txt:
    #             save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
    #         if save_json:
    #             save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
    #         callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

    #     # Plot images
    #     if plots and batch_i < 5:
    #         plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
    #         plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

    #     callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)
    # end_time=time.time()
    # LOGGER.info(f'Inference time: {end_time-start_time}')

    # # Compute metrics
    # stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # if len(stats) and stats[0].any():
    #     tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
    #     ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    #     mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    # nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # # Print results
    # pf = '%22s' + '%11i' * 2 + '%11.4g' * 4  # print format
    # LOGGER.info(pf % ('all', seen, nt.sum(), mp*100, mr*100, map50*100, map*100))
    # if nt.sum() == 0:
    #     LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # # Print results per class
    # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # # Print speeds
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # if not training:
    #     shape = (batch_size, 3, imgsz, imgsz)
    #     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # # Plots
    # if plots:
    #     confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    #     callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    


def parse_opt():
    parser = argparse.ArgumentParser()
    # retrain: 'retrained_best.pt'
    # original: 'best.pt'
    parser.add_argument('--data', type=str, default=ROOT / 'data/patch.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'retrained_best.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/NAPGuard_val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--NFSI',default=1, action='store_true', help='using NFSI strategy')
    parser.add_argument('--thres_factor',type=float, default=2.0, help='threshold factor')
    parser.add_argument('--sigma',type=float, default=3.0, help='standard variation of Gaussian kernel')

    parser.add_argument('--advdataset_path',type=str, default='/data/zjh/code/eval/data/xm_eval/adv_imgs/T-SEA')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('patch.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
