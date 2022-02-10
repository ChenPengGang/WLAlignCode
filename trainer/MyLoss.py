import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def Other_label(labels,num_classes):
    index=torch.randint(num_classes, (labels.shape[0],)).to(labels.device)
    other_labels=labels+index
    other_labels[other_labels >= num_classes]=other_labels[other_labels >= num_classes]-num_classes
    return other_labels


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,self_in_vectors,self_out_vectors, target_input_vectors, source_output_vectors,
                noise_vectors_self,noise_vectors_input,noise_vectors_output,
                noise_vectors_self2, noise_vectors_input2, noise_vectors_output2):
        BATCH_SIZE, embed_size = target_input_vectors.shape

        #计算Object1=======================================================================================
        #将输入词向量与目标词向量作维度转化处理
        self_in_vectors_T=self_in_vectors.view(BATCH_SIZE,embed_size,1)
        input_vectors_I=target_input_vectors.view(BATCH_SIZE,1,embed_size)
        output_vectors_I=source_output_vectors.view(BATCH_SIZE,1,embed_size)
        self_out_vectors_T=self_out_vectors.view(BATCH_SIZE,embed_size,1)

        input_vectors_T = target_input_vectors.view(BATCH_SIZE, embed_size, 1)
        output_vectors_T = source_output_vectors.view(BATCH_SIZE, embed_size, 1)

        #目标词损失
        p1_loss=torch.bmm(input_vectors_I,self_in_vectors_T).sigmoid().log()
        p2_loss=torch.bmm(output_vectors_I,self_out_vectors_T).sigmoid().log()


        #负样本损失
        p1_noise_loss = torch.bmm(noise_vectors_input.neg() ,self_in_vectors_T).sigmoid().log()
        p2_nose_loss = torch.bmm(noise_vectors_self.neg(),output_vectors_T).sigmoid().log()
        p1_noise_loss = p1_noise_loss.squeeze().sum(1)
        p2_nose_loss=p2_nose_loss.squeeze().sum(1)

        p1_noise_loss_ex = torch.bmm(noise_vectors_self.neg(), input_vectors_T).sigmoid().log()
        p2_nose_loss_ex = torch.bmm(noise_vectors_output.neg(), self_out_vectors_T).sigmoid().log()
        p1_noise_loss_ex = p1_noise_loss_ex.squeeze().sum(1)
        p2_nose_loss_ex = p2_nose_loss_ex.squeeze().sum(1)

        #计算Object2========================================================================================
        p1_noise_loss2 = torch.bmm(noise_vectors_input2.neg(), self_in_vectors_T).sigmoid().log()
        p2_nose_loss2 = torch.bmm(noise_vectors_self2.neg(), output_vectors_T).sigmoid().log()
        p1_noise_loss2 = p1_noise_loss2.squeeze().sum(1)
        p2_nose_loss2 = p2_nose_loss2.squeeze().sum(1)

        p1_noise_loss_ex2 = torch.bmm(noise_vectors_self2.neg(), input_vectors_T).sigmoid().log()
        p2_nose_loss_ex2 = torch.bmm(noise_vectors_output2.neg(), self_out_vectors_T).sigmoid().log()
        p1_noise_loss_ex2 = p1_noise_loss_ex2.squeeze().sum(1)
        p2_nose_loss_ex2 = p2_nose_loss_ex2.squeeze().sum(1)

        #综合计算两类损失
        return -(p1_loss+p2_loss+p1_noise_loss+p2_nose_loss+p1_noise_loss_ex+p2_nose_loss_ex+
                 p1_noise_loss2+p2_nose_loss2+p1_noise_loss_ex2+p2_nose_loss_ex2).mean()

class BCENeighborLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceLoss=nn.BCEWithLogitsLoss()
        self.mseLoss=nn.MSELoss()

    def forward(self,layers_label,layers_mark,sam_layers):
        loss_list=[]
        for mark,label in zip(layers_mark,layers_label):
            loss1=self.bceLoss(mark,label)
            loss2=self.mseLoss(mark,label)
            loss_list.append(loss1+loss2)
        loss=loss_list[0]
        for i in range(1,sam_layers-1):
            loss+=loss_list[i]
        return loss

class TripCenterLoss_margin(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True, device='cuda:1'):
        super(TripCenterLoss_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.device=device

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels,margin):
        other_labels = Other_label(labels, self.num_classes)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]
        other_labels = other_labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask_other = other_labels.eq(classes.expand(batch_size, self.num_classes))
        dist_other = distmat[mask_other]
        loss = torch.max(margin+dist-dist_other,torch.tensor(0.0).to(self.device)).sum() / batch_size
        return loss