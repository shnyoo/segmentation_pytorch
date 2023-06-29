import os
import sys
import time
import timeit
import torch
import pprint
import pathlib
import logging

import numpy as np
from torch import nn
import albumentations
from tqdm import tqdm
from models.hrnet import HRNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from utils.transformations import re_normalize
from utils.modelsummary import get_model_summary
from utils.runners import train, validate, testval, testvideo
from utils.data_utils import get_labels, label_mapping, SegmentationDataset, display
from utils.train_utils import AverageMeter, CrossEntropy, get_confusion_matrix, create_logger
from utils.transformation_pipelines import (get_transforms_training, get_transforms_validation, 
                                            get_transforms_evaluation, get_transforms_video)
from configs.hrnet_config import config as cfg
from utils.lr_schedule import LrUpdater,PolyLrUpdater

labels = get_labels()
id2label =      { label.id      : label for label in labels }
trainid2label = { label.trainId : label for label in labels }

def cityscapes_label_to_rgb(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for key, val in trainid2label.items():
        indices = mask == key
        mask_rgb[indices.squeeze()] = val.color 
    return mask_rgb

transforms_training = get_transforms_training(cfg)
transforms_validation = get_transforms_validation(cfg)
transforms_evaluation = get_transforms_evaluation(cfg)

train_dataset = SegmentationDataset(cfg = cfg.DATASET, split = "train", transform = transforms_training)
valid_dataset = SegmentationDataset(cfg = cfg.DATASET, split = "val", transform = transforms_validation)
eval_dataset = SegmentationDataset(cfg = cfg.DATASET, split = "val", transform = transforms_evaluation)


train_dataloader = DataLoader(dataset = train_dataset, batch_size = cfg.TRAIN.BATCH_SIZE, 
                              shuffle = True, num_workers = 0)
valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = cfg.TRAIN.BATCH_SIZE, 
                              shuffle = True, num_workers = 0)
eval_dataloader = DataLoader(dataset = eval_dataset, batch_size = 1, 
                             shuffle = False, num_workers = 0)

x, y, _, names = next(iter(train_dataloader))
xv, yv, _, vnames = next(iter(valid_dataloader))
xt, yt, _, tnames = next(iter(eval_dataloader))
#xvd, _, tnames = next(iter(video_dataloader))

x_min, x_max = x.min(), x.max()

idx = 0
#display([re_normalize(x[idx].permute(1,2,0).numpy()), cityscapes_label_to_rgb(y[idx])])
#display([re_normalize(xv[idx].permute(1,2,0).numpy()), cityscapes_label_to_rgb(yv[idx])])
#display([re_normalize(xt[idx].permute(1,2,0).numpy()), cityscapes_label_to_rgb(yt[idx])])



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = HRNet(cfg).to(device)


criterion = CrossEntropy(
    ignore_label=cfg.DATASET.IGNORE_LABEL, 
    weight=train_dataset.class_weights
)#.cuda()

optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=cfg.TRAIN.BASE_LR, 
    momentum=cfg.TRAIN.MOMENTUM, 
    weight_decay=cfg.TRAIN.WD
)

# model.init_weights(pretrained = cfg.MODEL.PRETRAINED)

#model.load_state_dict(torch.load("weights/hrnet_w48.pth", map_location=torch.device('cpu')))
model.eval()



lr_scheduler=PolyLrUpdater(max_iters=cfg.TRAIN.DECAY_STEPS,optimizer=optimizer,epoch_len=cfg.TRAIN.EPOCHS)

def run_train_loop():
    
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, 
        cfg_name="seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch484", 
        phase='train'
    )
    
    # dump_input = torch.rand((1, 3, 512, 1024))
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    best_mIoU = 0
    
    start = timeit.default_timer()
    
    for epoch in range(cfg.TRAIN.EPOCHS):
        
        train(
            cfg=cfg, 
            dataloader=train_dataloader, 
            model=model, 
            loss_fn=criterion, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            epoch=epoch, 
            scaler=torch.cuda.amp.GradScaler(),
            writer_dict=writer_dict
          
        )
        
        print(epoch,"epoch train finish")

        valid_loss, mean_IoU, IoU_array = validate(
            cfg=cfg, 
            dataloader=valid_dataloader, 
            model=model,  
            loss_fn=criterion,
            writer_dict=writer_dict,
            trainid2label=trainid2label
        
        )
        
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.state_dict(), os.path.join(final_output_dir, 'best.pth'))

        msg = 'Epoch {}/{} --- Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f} \n'.format(
            epoch+1, cfg.TRAIN.EPOCHS, valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        
    torch.save(model.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int((end-start)/3600))
    logger.info('Done')

run_train_loop()

#testvideo(config, video_dataloader, model, sv_dir='outputs', sv_pred=True)


print("mean IoU: {:.3f}, mean Accuracy: {:.3f}, Pixel Accuracy: {:.3f}".format(mean_IoU, mean_acc, pixel_acc))



#for key, val in trainid2label.items():
#    if key != config['IGNORE_LABEL'] and key != -1:
#        print("{} --- IoU: {:.2f}".format(val.name, IoU_array[key]))

