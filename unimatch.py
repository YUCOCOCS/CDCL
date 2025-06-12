import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.nn.functional as F
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
import megengine.functional as FF
import torch.distributed as dist
import numpy as np
import time

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation with Confidence-Driven Consistency Learning')
parser.add_argument('--config', type=str, default='/1.8T/user/New_sota/UniMatch-main/configs/pascal.yaml')
parser.add_argument('--labeled-id-path', type=str, default='/1.8T/user/New_sota/UniMatch-main/splits/pascal/1464/labeled.txt')
parser.add_argument('--unlabeled-id-path', type=str, default='/1.8T/user/New_sota/UniMatch-main/splits/pascal/1464/unlabeled.txt')
parser.add_argument('--save-path', type=str, default='/1.8T/user/New_sota/UniMatch-main/VOC_Result/Test')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

#####计算低置信度像素所占的比例

def nl_em_loss(pred_s, pseudo_label, k, mask_pred, ignore_mask,p_cutoff):
    B,C,H,W = pseudo_label.size() # B C H W
    softmax_pred = pred_s.softmax(dim=1)  #获得强增强视图的预测结果
    topk = torch.topk(pseudo_label, k,dim=1)[1]  # 返回每个像素位置的K个最大类概率的索引 B k  H W 
    mask_k = torch.ones_like(pseudo_label).to(torch.int64) # B C H W
    mask_k.scatter_(1, topk, 0)
    # print(mask_k.size(),mask_k.sum())
    # mask_k.scatter(1,topk,torch.zeros_like(topk).to(torch.int64))  #将高置信度区域置为0, B C H W
    label = pseudo_label.argmax(dim=1)  # 标签 B H W 
    #### 将伪标签的位置置为 1
    # mask_k.scatter(1,label.unsqueeze(1).expand(-1,C,-1,-1),torch.ones_like(label.unsqueeze(1).expand(-1,C,-1,-1)).to(torch.int64))
    mask_k.scatter_(1,label.unsqueeze(1),torch.ones_like(label.unsqueeze(1)).to(torch.int64))
    result = torch.where(mask_k == 1, softmax_pred, torch.zeros_like(softmax_pred))  # 对应的 伪标签和 低置信度区域的概率 B C H W
    result = result.contiguous().permute(0,2,3,1).reshape(-1,C).sum(dim=1) # 变成 (B H W )
    processed_tensor = (1-result+1e-7)/(k-1)
    soft_ml = processed_tensor.unsqueeze(1).expand(-1,C)
    mask = 1 - mask_k   # B C H W
    mask = mask.contiguous().permute(0,2,3,1).reshape(-1,C)  # (B H W) C
    mask_really = mask_pred.view(-1,1) * mask
    softmax_pred = softmax_pred.permute(0,2,3,1).reshape(-1,C)
    mask_really = torch.where((mask_really==1)&(softmax_pred>p_cutoff**2), torch.zeros_like(mask_really), mask_really)  # 掩码就是阈值高，但是不是真实标签的类别
    loss_em = -(soft_ml*torch.log(softmax_pred+1e-10)+(1-soft_ml)*torch.log(1-softmax_pred+1e-10))
    loss_em = (loss_em * mask).sum()/(mask.sum()+1e-10)
    return loss_em


def compute_complement_loss(P_weak, P_strong, ignore_mask,threshold=0.95, k=12):
    """
    计算互补性损失，用于传递排除类别的知识。

    参数:
        - P_weak: 弱增强视图的预测概率, 形状 (B, C, H, W)
        - P_strong: 强增强视图的预测概率, 形状 (B, C, H, W)
        - threshold: 置信度阈值，用于筛选低置信度区域
        - k: 排除类别的数量，选择低置信度像素中置信度最低的 k 个类别

    返回:
        - complement_loss: 互补性损失
    """
    # 1. 获取最高置信度及其类别索引
    max_probs, _ = torch.max(P_weak, dim=1)  # (B, H, W)
    
    # 2. 生成低置信度掩码
    low_confidence_mask = (max_probs < threshold) * (ignore_mask!=255)  # (B, H, W)
    
    # 3. 在低置信度区域排序每个像素的类别概率，选取排除类别
    B, C, H, W = P_weak.shape
    P_weak_sorted, sorted_indices = torch.sort(P_weak, dim=1, descending=False)  # 从低到高排序
    exclude_classes = sorted_indices[:, :k, :, :]  # (B, k, H, W)
    
    # 4. 计算互补性损失
    # 在强增强视图中，降低排除类别的概率响应
    complement_loss = 0.0
    for i in range(k):
        exclude_class = exclude_classes[:, i, :, :]  # (B, H, W)
        # Gather强增强视图在排除类别上的概率
        exclude_prob = torch.gather(P_strong, dim=1, index=exclude_class.unsqueeze(1)).squeeze(1)  # (B, H, W)
        
        # 计算互补损失，仅在低置信度掩码的区域应用
        complement_loss += -torch.log(1 - exclude_prob + 1e-7) * low_confidence_mask  # 防止 log(0)
    
    # 5. 求互补损失的平均值
    complement_loss = complement_loss.mean()
    return complement_loss




def reduce_tensor(tensor, mean=True):
    dist.all_reduce(tensor)
    if mean:
        return tensor / dist.get_world_size()
    return tensor 


def cal_topK(pred_u_s1,conf_u_w_probability_copy1,cfg):
    max_probs, max_idx = torch.max(conf_u_w_probability_copy1,dim=1)
    maxk = cfg['nclass']
    batch_size, C ,h, w = pred_u_s1.shape
    target = max_idx.view(-1)
    pred_u_s1_copy = pred_u_s1.permute(0,2,3,1)
    pred_u_s1_copy = pred_u_s1_copy.contiguous().view(-1,C)  # 尺寸变为B × C
    _,pred = torch.topk(pred_u_s1_copy,maxk)
    pred = pred.permute(1,0)
    # correct = FF.equal(pred,torch.broadcast_to(target.reshape(1,-1),pred.shape)).to(torch.float32)
    correct = (pred == target.unsqueeze(0).expand(pred.size())).to(torch.float32)
    top_k = -1
    for k in list(np.arange(2,cfg['nclass']+1)):
        correct_k = correct[:k].reshape(-1).sum(0)
        acc_single = torch.mul(correct_k,100./(batch_size*h*w))
        acc_parallel = reduce_tensor(acc_single)
        if acc_parallel > 99.99:
            top_k = k
            break
    return top_k
    

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_loss_em = AverageMeter()
        total_loss_last = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_high_confidence_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            image_u_strong_1 = img_u_s1.clone()
            image_u_strong_2 = img_u_s2.clone()

            start_time = time.time()

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix_probability = F.softmax(pred_u_w_mix,dim=1)
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w_probability = F.softmax(pred_u_w,dim=1)  # B C H W
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            pred_u_s1_proba = F.softmax(pred_u_s1,dim=1)
            pred_u_s2_proba = F.softmax(pred_u_s2,dim=1)

            cutmix_box1_3_d = cutmix_box1.unsqueeze(1).repeat(1,cfg['nclass'],1,1)
            cutmix_box2_3_d = cutmix_box2.unsqueeze(1).repeat(1,cfg['nclass'],1,1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            
            conf_u_w_mix_probability_copy1 = conf_u_w_mix_probability.clone()
            conf_u_w_mix_probability_copy2 = conf_u_w_mix_probability.clone()
            conf_u_w_probability_copy1 = conf_u_w_probability.clone()
            conf_u_w_probability_copy2 = conf_u_w_probability.clone()

            conf_u_w_probability_copy1[cutmix_box1_3_d == 1]  = conf_u_w_mix_probability_copy1[cutmix_box1_3_d == 1]
            conf_u_w_probability_copy2[cutmix_box2_3_d == 1] = conf_u_w_mix_probability_copy2[cutmix_box2_3_d == 1]

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]


            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            
            B, H , W = conf_u_w_cutmixed1.size()
            current_high_ratio = (conf_u_w_cutmixed1 >= cfg['conf_thresh']).sum().item() / (B * H * W)
            total_high_confidence_ratio.update(current_high_ratio)

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            select = torch.ge(conf_u_w_cutmixed1, cfg['conf_thresh']).to(torch.int)
            k_value_1 = cal_topK(pred_u_s1, conf_u_w_probability_copy1,cfg)  # 返回Top ——K
            loss_em_1 = nl_em_loss(pred_u_s1, conf_u_w_probability_copy1, k_value_1, select,ignore_mask_cutmixed1.clone(), cfg['conf_thresh'])


            select_2 = torch.ge(conf_u_w_cutmixed2, cfg['conf_thresh']).to(torch.int)
            k_value_2 = cal_topK(pred_u_s2, conf_u_w_probability_copy2,cfg)  # 返回Top ——K
            loss_em_2 = nl_em_loss(pred_u_s2, conf_u_w_probability_copy2, k_value_2, select_2,ignore_mask_cutmixed2.clone(), cfg['conf_thresh'])

            # print(loss_em_1,loss_em_2)

            loss_em = cfg['em_loss_weight'] * (loss_em_1 + loss_em_2) / 2.0

            loss_last_1 = compute_complement_loss(conf_u_w_probability_copy1, pred_u_s1_proba,ignore_mask_cutmixed1)
            loss_last_2 = compute_complement_loss(conf_u_w_probability_copy2, pred_u_s2_proba,ignore_mask_cutmixed2)

            loss_last = cfg['loss_last'] * (loss_last_1 + loss_last_2) / 2.0

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0    + loss_em

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_em.update(loss_em.item())
            total_loss_last.update(loss_last.item())
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            end_time = time.time()
            execution_time = end_time - start_time
            

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Loss_em: {:.6f} Loss_last: {:.10f} Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_loss_em.avg, total_loss_last.avg ,total_mask_ratio.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        print("当前的高置信度像素的比例为: %s"%(total_high_confidence_ratio.avg))

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
