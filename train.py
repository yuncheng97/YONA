import os
import sys
import cv2
import argparse
import random
import time
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
sys.dont_write_bytecode = True
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from model import YONA
from data import TrainData
from utils import focal_loss, reg_l1_loss, ModelEma
from utils import decode_bbox, postprocess, init_distributed_mode

class Train:
    def __init__(self, args, exp_path):
        self.args         = args
        self.logger       = SummaryWriter(exp_path)
        ## data
        train_dataset     = TrainData(args)
        sampler           = DistributedSampler(train_dataset)
        self.loader       = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, sampler=sampler)
        ## model
        self.model        = YONA(args).cuda()
        self.model        = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        
        self.model.train()
        ## optimizer
        self.optimizer    = torch.optim.AdamW(self.model.parameters(), args.lr, weight_decay=5e-4)
        ## learning rate scheduler
        if self.args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [40], gamma=0.1, last_epoch=-1, verbose=False)
        elif self.args.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=60, eta_min=1e-5)
        elif self.args.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5, last_epoch=-1, verbose=False)

        ## model ema
        self.ema = ModelEma(self.model, decay=0.9998)


    def compute_contrast_loss(self, feature, masks, temperature=0.2):
        """
        Compute the contrastive loss for the given feature and masks.

        Args:
            feature (torch.Tensor): The feature tensor.
            masks (torch.Tensor): The masks tensor.
            temperature (float, optional): Temperature for the contrastive loss. Defaults to 0.2.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        masks       = masks.unsqueeze(1)  # mask : [B, 1, H, W]
        pos         = (feature * masks).sum(dim=(2, 3)) / (masks.sum(dim=(2, 3)) + 1e-6)
        neg         = (feature * (1 - masks)).sum(dim=(2, 3)) / ((1 - masks).sum(dim=(2, 3)) + 1e-6)

        pos         = F.normalize(pos, dim=1)
        neg         = F.normalize(neg, dim=1)
        pos_neg     = torch.mm(pos, neg.transpose(1, 0))  # [B, B]
        pos_pos     = (pos * pos[torch.randperm(pos.size()[0])]).sum(dim=1)  # [B, 1]

        pos_logits  = torch.exp(pos_pos / temperature)
        neg_logits  = torch.exp(pos_neg / temperature)
        nce_loss    = (-torch.log(pos_logits / (pos_logits + neg_logits.sum(dim=1)))).mean(dim=0)

        return nce_loss

    def train(self):
        best_acc    = 0
        global_step = 0
        valer       = Validation(self.args.data_path, self.args.test_clips)
        for epoch in range(1, args.epoch):
            self.model.train()
            if epoch<2:
                for param in self.model.module.backbone.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.module.backbone.parameters():
                    param.requires_grad = True
                
            for i, (images, hms, whs, regs, reg_masks, masks) in enumerate(self.loader):
                images, hms, whs, regs, reg_masks, masks = images.cuda(), hms.cuda(), whs.cuda(), regs.cuda(), reg_masks.cuda(), masks.cuda()
                pre_feat, cur_feat, hm, wh, offset  = self.model(images, masks)
                
                with autocast():
                    contrast_loss   = self.compute_contrast_loss(torch.cat((cur_feat, pre_feat), dim=0), torch.cat((masks[:,0,:,:], masks[:,1,:,:]), dim=0))
                    c_loss          = focal_loss(hm, hms)
                    wh_loss         = reg_l1_loss(wh, whs, reg_masks)
                    off_loss        = reg_l1_loss(offset, regs, reg_masks)
                    loss            = c_loss + wh_loss*0.1 + off_loss + 0.3 * contrast_loss

                self.optimizer.zero_grad()  
                loss.backward()
                self.optimizer.step()
                self.ema.update(self.model) 

                global_step += 1
                self.logger.add_scalar('lr'   , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalars('loss', {'loss':loss.item(), 'c_loss':c_loss.item(), 'wh_loss':wh_loss.item(), 'off_loss':off_loss.item()}, global_step=global_step)
                if i % 100 == 0:
                    print(f'{datetime.now()} Epoch={epoch:03d}/{self.args.epoch:03d}, Step={i:04d}/{len(self.loader):04d}, loss={loss.item():0.4f}, c_loss={c_loss.item():0.4f}, wh_loss={wh_loss.item():0.4f}, off_loss={off_loss.item():0.4f}')
                    logging.info(f'{datetime.now()} Epoch={epoch:03d}/{self.args.epoch:03d}, Step={i:04d}/{len(self.loader):04d}, loss={loss.item():0.4f}, c_loss={c_loss.item():0.4f}, wh_loss={wh_loss.item():0.4f}, off_loss={off_loss.item():0.4f}')
            precision, recall, f1, f2 = valer.val(self.model, epoch)
            self.logger.add_scalar('Metrics/Precision', precision, global_step=global_step)
            self.logger.add_scalar('Metrics/Recall'   , recall   , global_step=global_step)
            self.logger.add_scalar('Metrics/F1-score' , f1       , global_step=global_step)
            self.logger.add_scalar('Metrics/F2-score' , f2       , global_step=global_step)
            if f1 > best_acc:
                torch.save(self.model.state_dict(), exp_path + '/'+'best.pth')
                print(f"saved best model at epoch {epoch}, f1 score: {f1:.3f}")
                logging.info(f"saved best model at epoch {epoch}, f1 score: {f1:.3f}")
                best_acc = f1
            self.scheduler.step()


def num_iou(bboxs, gt_bboxs):
    num_tp = 0
    for box in bboxs:
        flag = False
        for gt_box in gt_bboxs:
            xmin, ymin, xmax, ymax = box
            x1, y1, x2, y2         = gt_box
            width, height          = max(min(xmax, x2)-max(xmin, x1), 0), max(min(ymax, y2)-max(ymin, y1), 0)
            union                  = (xmax-xmin)*(ymax-ymin)+(x2-x1)*(y2-y1)
            inter                  = width*height
            iou                    = inter/(union-inter+1e-6)
            if iou>0.5 and width>0 and height>0:       
                flag = True
                break
        if flag:
            num_tp += 1
    return num_tp, len(bboxs)-num_tp, len(bboxs), len(gt_bboxs)


class Validation:
    def __init__(self, data_path, test_clips, test_mode = False):
        self.data_path  = data_path
        self.test_clips = test_clips
        self.mean       = np.array([0.485, 0.456, 0.406])
        self.std        = np.array([0.229, 0.224, 0.225])
        self.confidence = 0.3
        self.nms_iou    = 0.3
        ## data
        self.names      = []
        self.samples    = []
        self.mem_bank   = []
        self.offset     = []
        with open(self.data_path+'/test_box.txt') as test_box:
            self.lines = test_box.readlines()
            start = True
            for cur_idx in range(len(self.lines)):
                batch_clips = []
                sample     = self._load_sample(cur_idx)
                batch_clips.append(sample)
                name, boxs = sample
                if cur_idx == 0:
                    pre_idx = cur_idx
                else:
                    pre_idx = cur_idx - 1 
                sample = self._load_sample(pre_idx)
                batch_clips.append(sample)

                self.samples.append(batch_clips)
        print('testing samples:', len(self.samples))

        # loading name, images and bboxes
        print("Loading testing dataset:")
        self.Image  = []
        self.Bbox   = []
        self.Name   = []
        self.Height = []
        self.Width  = []
        for idx in range(len(self.samples)):
            image = torch.zeros((1, self.test_clips, 3, 512, 512))
            for clip in range(self.test_clips):
                name, bbox = self.samples[idx][clip]
                if 'CVC' in self.data_path:        
                    img   = cv2.imread(self.data_path+'/validation/'+name)
                elif 'SUN' in self.data_path:
                    image = cv2.imread(self.data_path+'/TestHardDataset/Frame/'+name)
                else:
                    image = cv2.imread(self.data_path+'/Test/Images/'+name)
                img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H,W,C    = img.shape
                img      = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)/255.0
                img      = (img-self.mean)/self.std
                img      = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda().float()
                image[:, clip, :, :, :] = img
                if clip == 0:
                    self.Name.append(name)
                    self.Bbox.append(bbox)
                    self.Height.append(H)
                    self.Width.append(W)
            self.Image.append(image)
            if idx   == int(len(self.samples) * 0.3):
                print('30%|'+'*'*15+' | '+str(idx)+'/'+str(len(self.samples)))
            elif idx == int(len(self.samples) * 0.6):
                print('60%|'+'*'*30+' | '+str(idx)+'/'+str(len(self.samples)))
        print('100%|'+'*'*50+' | '+str(idx)+'/'+str(len(self.samples)))
    def _load_sample(self, idx):
        name, boxs = self.lines[idx].strip().split(';')
        boxs       = boxs.split(' ')
        bbox       = []
        for i in range(len(boxs)//4):
            xmin, ymin, xmax, ymax = boxs[4*i:4*(i+1)]
            bbox.append([max(int(xmin),0), max(int(ymin),0), int(xmax), int(ymax)])
        return [name, bbox]


    def _en_queue(self, bbox):
        self.mem_bank.append(bbox)

    def _de_queue(self, idx):
        masks   = torch.zeros((1, self.test_clips, 512, 512)).cuda()
        if idx - 1 < 0:
            masks[0, 1, :, :] = 0
        else:
            if self.mem_bank[idx-1] == 0:
                masks[0, 1, :, :] = 0
            else:
                bboxes = self.mem_bank[idx-1]
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax = bbox[:4]
                    masks[0, 1, int(ymin):int(ymax), int(xmin):int(xmax)] = 1
        return masks

    def val(self, model, epoch):
        model.eval()
        with torch.no_grad():
            num_tps, num_fps, num_dets, num_gts = 0, 0, 0, 0
            seconds = 0
            for idx in range(len(self.samples)):
                images     = self.Image[idx].cuda()
                gt_bbox    = self.Bbox[idx]
                masks      = self._de_queue(idx)
                masks      = masks.cuda()
                start = time.time()
                _, _, heatmap, whpred, offset   = model(images, masks)
                end = time.time()
                seconds += end - start
                H          = self.Height[idx]
                W          = self.Width[idx]
                outputs    = decode_bbox(heatmap, whpred, offset, self.confidence)
                results    = postprocess(outputs, (H,W), self.nms_iou)
                if results[0] is None:
                    num_gts += len(gt_bbox)
                    self._en_queue(0)
                    continue
                confidence = results[0][:, 4]
                bboxs      = []
                confident_bboxs = []
                for result in results[0]:
                    box                     = result[:4]
                    confidence              = result[4]
                    ymin, xmin, ymax, xmax  = box
                    xmin, ymin, xmax, ymax  = int(xmin), int(ymin), int(xmax), int(ymax)
                    bboxs.append([xmin, ymin, xmax, ymax])
                    if confidence >= 0:
                        confident_bboxs.append([xmin, ymin, xmax, ymax])
                if confident_bboxs == []:
                    self._en_queue(0)
                else:         
                    self._en_queue(bboxs)
                num_tp, num_fp, num_det, num_gt = num_iou(bboxs, gt_bbox)
                num_tps, num_fps, num_dets, num_gts = num_tps+num_tp, num_fps+num_fp, num_dets+num_det, num_gts+num_gt 

        fps  = len(self.samples) / seconds
        precision = num_tps / (num_dets+1e-6)
        recall    = num_tps / num_gts
        f1        = 2*num_tps/(num_dets+num_gts)
        f2        = (5*precision*recall) / (4*precision + recall+1e-6) 
        tables  = [[epoch, num_tps, num_fps, num_dets, num_gts, precision, recall, f1, f2, fps]]
        headers = ['epoch', 'num_tps', 'num_fps', 'num_dets', 'num_gts', 'precision' ,'recall', 'f1', 'f2', 'fps']
        print(tabulate(tables, headers, tablefmt="grid", numalign="center"))
        logging.info(tabulate('\n'+tables, headers, tablefmt="grid", numalign="center"))
        return precision, recall, f1, f2 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone'   , type=str   , default='pvt_v2_b2'              )
    parser.add_argument('--snapshot'   , type=str   , default=None                     )
    parser.add_argument('--epoch'      , type=int   , default=20                       )
    parser.add_argument('--lr'         , type=float , default=1e-4                     )
    parser.add_argument('--scheduler'  , type=str   , default='step'                   )
    parser.add_argument('--batch_size' , type=int   , default=16                       )
    parser.add_argument('--train_clips', type=int   , default=2                        )
    parser.add_argument('--test_clips' , type=int   , default=2                        )
    parser.add_argument('--n_threads'  , type=int   , default=6                        )
    parser.add_argument('--data_path'  , type=str   , default='../dataset/'            )
    parser.add_argument('--save_path' , type=str   , default='./result'                )
    parser.add_argument('--model_name' , type=str   , default='YONA'            )
    parser.add_argument('--gpu_id'     , type=str   , default='0'                      )
    # distribution setting
    parser.add_argument('--distributed', action='store_true', help='use distribution training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    ## experiment path ##
    save_path          = os.path.join(args.save_path, args.model_name)
    current_timestamp  = datetime.now().timestamp()
    current_datetime   = datetime.fromtimestamp(current_timestamp+29220)  # different time zone
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    exp_path           = os.path.join(save_path, 'log_' + formatted_datetime)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(exp_path, exist_ok=True)

    init_distributed_mode(args)
    trainer = Train(args, exp_path)

    logging.basicConfig(filename=exp_path+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Config")
    table = [[args.model_name, args.backbone, args.epoch, args.batch_size, args.lr, args.scheduler, args.train_clips, args.test_clips, torch.cuda.get_device_name(0), "use wei (not 1-wei) as cosine weight"]]
    headers = ["Model", "Backbone", "Epoch", "Batch", "LR", "Scheduler", "Train Clips", "Test Clips", "GPU", "Note"]
    print(tabulate(table, headers, tablefmt="grid", numalign="center"))
    logging.info('\n'+tabulate(table, headers, tablefmt="grid", numalign="center"))
    start_time = time.time()

    trainer.train()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training completed.\nTotal training time: {format(total_time_str)}')
    logging.info(f'Training completed.\nTotal training time: {format(total_time_str)}')
