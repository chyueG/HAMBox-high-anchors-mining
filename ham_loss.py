import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.bbox_utils import *

class HAMLoss(nn.Module):
    def __init__(self,num_cls,variance, use_gpu = True):
        super(HAMLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_cls
        self.variance = variance

        #self.variance = [0.1, 0.2]
        self.K = 5
        self.T1 = 0.35
        self.T2 = 0.5
        self.alpha = 0.25
        self.gamma = 2
        self.smoothl1loss = SmoothL1Loss()

    def forward(self, predictions, targets, im_names):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loss_cls,loss_loc = list(),list()
        loc_data, conf_data, priors = predictions
        priors = priors
        num = loc_data.size(0)

        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors
            conf     = conf_data[idx]
            loc       = loc_data[idx]

            if truths.shape[0] == 0:
                if torch.cuda.is_available():
                    loss_cls.append(torch.tensor(0).float().cuda())
                    loss_loc.append(torch.tensor(0).float().cuda())
                else:
                    loss_cls.append(torch.tensor(0).float())
                    loss_loc.append(torch.tensor(0).float())

                continue
            iou = jaccard(truths, point_form(defaults))  ###### gt_boxes.size()[0]*anchors.size()[0]

            best_truth_score, best_truth_idx = iou.max(0, keepdim=True)
            best_truth_score.squeeze_(0)
            best_truth_idx.squeeze_(0)

            assign_annotation = truths[best_truth_idx]

            n_pos_idx = best_truth_score.gt(self.T1)

            match_labels = labels[best_truth_idx]
            n_encoded_loc = encode(assign_annotation,priors,self.variance)

            m_anchor_cnt = torch.bincount(best_truth_idx[n_pos_idx])  ######r
            match_anchor_cnt = torch.zeros_like(labels)
            match_anchor_cnt[:m_anchor_cnt.size(0)] = m_anchor_cnt
            normal_targets = -1*torch.ones_like(conf).cuda()
            normal_targets[n_pos_idx,:] = 0
            normal_targets[n_pos_idx,match_labels[n_pos_idx].long()] = 1
            decoded_regress = decode(loc, defaults, self.variance)

            c_iou = jaccard(truths, decoded_regress)

            if torch.isnan(c_iou).any():
                print("check decoded_regress")

            c_best_truth_score, c_best_truth_idx = c_iou.max(0, keepdim=True)
            c_best_truth_score.squeeze_(0)
            c_best_truth_idx.squeeze_(0)

            ignore_idx = torch.bitwise_and(c_best_truth_score.gt(self.T2), best_truth_score.lt(self.T1))
            neg_idx    =  ~torch.bitwise_or(n_pos_idx,ignore_idx)
            normal_targets[neg_idx,:] = 0
            #normal_targets[neg_idx,0] = 1

            if n_pos_idx.sum() > 0:
                loss_cls_normal = focal_loss(normal_targets, conf, self.alpha, self.gamma, None)
                loss_cls_normal = torch.where(normal_targets.ne(-1),loss_cls_normal,torch.tensor(0.).float().cuda())
                loss_cls_normal = loss_cls_normal.sum() / n_pos_idx.sum()
                loc_normal = loc[n_pos_idx]
                assign_loc = n_encoded_loc[n_pos_idx]
                loss_loc_normal = self.smoothl1loss(loc_normal, assign_loc)

            else:
                loss_cls_normal = torch.tensor(0.).cuda()
                loss_loc_normal = torch.tensor(0.).cuda()

            outface_idx_mask = match_anchor_cnt.lt(self.K)
            loss_cls_com = torch.tensor(0.).cuda()
            loss_loc_com = torch.tensor(0.).cuda()

            if outface_idx_mask.any():
                outface_idx     = torch.arange(0,truths.size(0))[outface_idx_mask]
                c_anchor_num = (self.K - match_anchor_cnt)[outface_idx_mask]
                ####### mask outerface already match anchors in first step

                for _,value in enumerate(outface_idx):
                     c_mask_idx = torch.bitwise_and(best_truth_idx.eq(value),best_truth_score.gt(self.T1))
                     c_mask_idx = torch.arange(0,defaults.size(0))[c_mask_idx]
                     c_iou[value.long()].index_fill_(0,c_mask_idx,torch.tensor(0.).cuda())

                outface_iou = c_iou[outface_idx_mask]
                c_best_truth_score, c_best_truth_idx = outface_iou.max(0, keepdim=True)
                c_best_truth_idx.squeeze_(0)
                c_best_truth_score.squeeze_(0)

                c_match_nums = torch.zeros(c_anchor_num.shape)
                c_match_nums = c_match_nums.type_as(c_best_truth_idx)

                c_match_nums = c_match_nums.scatter(0,torch.arange(len(torch.bincount(c_best_truth_idx[c_best_truth_score.gt(self.T2)]))),\
                                     torch.bincount(c_best_truth_idx[c_best_truth_score.gt(self.T2)]))
                c_anchor_num = torch.where(c_match_nums.lt(c_anchor_num.long()),c_match_nums,c_anchor_num.long())

                if c_anchor_num.sum().gt(0):
                    com_targets = -1 * torch.ones_like(conf).cuda()
                    c_iou_score, c_iou_idx = outface_iou.sort(dim=1, descending=True)
                    com_iou = torch.zeros_like(best_truth_idx).float().cuda()
                    com_pos_idx = torch.zeros_like(best_truth_idx).cuda()
                    for outer_idx, value in enumerate(outface_idx):
                        if c_anchor_num[outer_idx].gt(0):
                            com_pos_idx.index_fill_(0,c_iou_idx[outer_idx,:c_anchor_num[outer_idx].long()],torch.tensor(1).cuda())
                            c_best_truth_idx.index_fill_(0,c_iou_idx[outer_idx,:c_anchor_num[outer_idx].long()],value)
                            com_iou[c_iou_idx[outer_idx,:c_anchor_num[outer_idx].long()]] = c_iou_score[outer_idx,:c_anchor_num[outer_idx].long()]

                            # com_targets.index_put_(c_anchor_box_axis, labels[i])
                            com_targets[c_iou_idx[outer_idx,:c_anchor_num[outer_idx].long()], :] = 0
                            com_targets[c_iou_idx[outer_idx,:c_anchor_num[outer_idx].long()], labels[value].long()] = labels[value].half()
                        else:
                            continue
                    com_pos_idx = com_pos_idx.bool()
                    c_encoded_loc = encode(truths[c_best_truth_idx], priors, self.variance)
                    c_assign_annotations = c_encoded_loc[com_pos_idx]
                    loc_c = loc[com_pos_idx]
                    loss_loc_com = self.smoothl1loss(loc_c,c_assign_annotations)
                    loss_cls_com = focal_loss(com_targets, conf, self.alpha, self.gamma, com_iou)
                    loss_cls_com = torch.where(com_targets.ne(-1),loss_cls_com,torch.tensor(0.).cuda())
                    loss_cls_com = loss_cls_com.sum() / c_anchor_num.sum()

            loss_cls.append(loss_cls_normal+loss_cls_com)
            loss_loc.append(loss_loc_normal+loss_loc_com)

        return torch.stack(loss_loc).mean(dim=0), torch.stack(loss_cls).mean(dim=0)



def focal_loss(target,prob,alpha,gamma,f_iou=None):

    ce = F.binary_cross_entropy_with_logits(prob, target, reduction="none")
    alpha = target*alpha + (1.-target)*(1- alpha)
    pt = torch.where(target==1,prob,1-prob)
    if f_iou is not None:
        alpha = f_iou.unsqueeze(dim=1)*alpha

    return  alpha*(1.-pt)**gamma*ce

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


class SmoothL1Loss(nn.Module):
    def __init__(self,beta=0.11):
        super().__init__()
        self.beta = 0.11

    def forward(self,pred,target):
        x = (pred - target).abs()
        l1 = x - 0.5*self.beta
        l2 = 0.5*x**2/self.beta
        return torch.sum(torch.where(x>=self.beta,l1,l2))/pred.size(0)




