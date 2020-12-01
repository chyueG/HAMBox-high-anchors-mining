import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import *


class HAMLoss(nn.Module):
    def __init__(self, use_gpu = True):
        super(HAMLoss, self).__init__()
        self.use_gpu = use_gpu
        # self.num_classes = num_cls
        # self.variance = variance

        self.variance = [0.1, 0.2]
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
        loc_data, conf_data, priors = predictions
        # priors = priors
        batch_size = loc_data.size(0)
        prior_size = priors.size(0)
        defaults = priors.data

        loc_t = conf_data.new(batch_size,prior_size,4).float()
        conf_t = conf_data.new(batch_size,prior_size).float()
        cloc_t = conf_data.new(batch_size,prior_size,4).float()
        cconf_t = conf_data.new(batch_size,prior_size).float()
        cc_score_t = conf_data.new(batch_size,prior_size).float()

        for idx in range(batch_size):
            truths = targets[idx][:, :4].data.float()
            labels = targets[idx][:, -1].data.float()
            # conf   = conf_data[idx]
            loc_pred    = loc_data[idx]
           
            iou = jaccard(truths, point_form(defaults))  ###### gt_boxes.size()[0]*anchors.size()[0]

            best_truth_score, best_truth_idx = iou.max(0, keepdim=True)

            best_truth_score.squeeze_(0)
            best_truth_idx.squeeze_(0)

            for _ in range(iou.size(0)):
                best_prior_score,best_prior_idx = iou.max(1,keepdim=True)
                j = best_prior_score.max(0)[1][0]
                i = best_prior_idx[j][0]
                iou[j,:] = -1
                iou[:,i] = -1
                best_truth_score[i] = best_prior_score.max(0)[0][0]
                best_truth_idx[i] = j

            matches_s1 = truths[best_truth_idx]

            conf_s1 = labels[best_truth_idx]  # face label is 1
            conf_s1[best_truth_score<self.T1] = 0

            loc_s1 = encode(matches_s1,defaults,self.variance)



            decoded_regress = decode(loc_pred.detach(), defaults, self.variance)
            c_iou = jaccard(truths, decoded_regress)
            c_best_prior_score ,c_best_prior_idx = c_iou.max(0,keepdim=True)
            c_best_prior_score.squeeze_(0)
            c_best_prior_idx.squeeze_(0)
            c_iou = c_iou*(~c_iou.lt(self.T2))#####keep overlap greater than T2
            c_prior_score = -1 * torch.zeros_like(c_best_prior_score)
            #with torch.no_grad():
            ctopk_prior_score,ctopk_prior_idx = torch.topk(c_iou,self.K,dim=1)
            #c_prior_idx = -1*torch.ones_like(best_truth_score)
            for i in range(c_iou.size(0)):
                 for j in range(self.K):
                     if conf_s1[ctopk_prior_idx[i,j]].lt(1) and ctopk_prior_score[i,j].gt(0):
                         c_best_prior_idx[ctopk_prior_idx[i,j]] = i
                         c_prior_score[ctopk_prior_idx[i,j]] = ctopk_prior_score[i,j]


            matches_s2 = truths[c_best_prior_idx]
            loc_s2 = encode(matches_s2,defaults,self.variance)
            conf_s2 = labels[c_best_prior_idx]
            conf_s2[c_prior_score.lt(self.T2)] = -1
            ignore_idx = best_truth_score.lt(self.T1)*(~c_best_prior_score.lt(self.T2))*c_prior_score.lt(self.T2)
            conf_s1[ignore_idx] = -1

            conf_t[idx] = conf_s1
            cconf_t[idx] = conf_s2
            loc_t[idx] = loc_s1
            cloc_t[idx] = loc_s2
            cc_score_t[idx] = c_prior_score

        if conf_t.gt(0).sum()>0 and cconf_t.gt(0).sum()<1 :
            loc_loss = self.smoothl1loss(loc_data[conf_t>0],loc_t[conf_t>0])
            loss_cls_s1 = focal_loss(conf_data, conf_t, self.alpha, self.gamma, None)
            return loc_loss,loss_cls_s1
        elif conf_t.gt(0).sum()>0 and cconf_t.gt(0).sum()>0:
            loc_loss = self.smoothl1loss(loc_data[conf_t > 0], loc_t[conf_t > 0]) + self.smoothl1loss(loc_data[cconf_t > 0], cloc_t[cconf_t > 0])
            loss_cls_s1 = focal_loss(conf_data, conf_t, self.alpha, self.gamma, None)
            loss_cls_s2 = focal_loss(conf_data, cconf_t, self.alpha, self.gamma, cc_score_t)
            cls_loss =  loss_cls_s1 + loss_cls_s2
            return loc_loss,cls_loss
        elif conf_t.gt(0).sum()<1 and cconf_t.gt(0).sum()>0:
            loc_loss = self.smoothl1loss(loc_data[cconf_t > 0], cloc_t[cconf_t > 0])
            loss_cls_s2 = focal_loss(conf_data, cconf_t, self.alpha, self.gamma, cc_score_t)
            return loc_loss, loss_cls_s2
        else:
            return torch.tensor(1e-4).cuda(),torch.tensor(1e-4).cuda()



def focal_loss(prob,target,alpha,gamma,f_iou=None):

    num_classes = prob.size(-1)
    target      = target.view(-1,1)
    keep = (target >= 0).float()
    target[target < 0] = 0  #
    pos_anchor_nums = target.gt(0).sum()

    prob        = prob.view(-1,num_classes)
    prob        = prob.gather(1,target.long())

    ce = F.binary_cross_entropy_with_logits(prob, target, reduction="none")
    alpha = target*alpha + (1.-target)*(1- alpha)
    pt = torch.where(target==1,torch.sigmoid(prob),1-torch.sigmoid(prob))
    if f_iou is not None:
        with torch.no_grad():
            f_iou = f_iou.view(-1,1)

            alpha = f_iou*alpha

    loss = alpha*(1.-pt)**gamma*ce
    loss = loss*keep
    loss = loss.sum()/pos_anchor_nums
    return loss


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





