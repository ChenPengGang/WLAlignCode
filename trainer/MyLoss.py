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

        self_in_vectors_T=self_in_vectors.view(BATCH_SIZE,embed_size,1)
        input_vectors_I=target_input_vectors.view(BATCH_SIZE,1,embed_size)
        output_vectors_I=source_output_vectors.view(BATCH_SIZE,1,embed_size)
        self_out_vectors_T=self_out_vectors.view(BATCH_SIZE,embed_size,1)

        input_vectors_T = target_input_vectors.view(BATCH_SIZE, embed_size, 1)
        output_vectors_T = source_output_vectors.view(BATCH_SIZE, embed_size, 1)

        p1_loss=torch.bmm(input_vectors_I,self_in_vectors_T).sigmoid().log()
        p2_loss=torch.bmm(output_vectors_I,self_out_vectors_T).sigmoid().log()

        p1_noise_loss = torch.bmm(noise_vectors_input.neg() ,self_in_vectors_T).sigmoid().log()
        p2_nose_loss = torch.bmm(noise_vectors_self.neg(),output_vectors_T).sigmoid().log()
        p1_noise_loss = p1_noise_loss.squeeze().sum(1)
        p2_nose_loss=p2_nose_loss.squeeze().sum(1)

        p1_noise_loss_ex = torch.bmm(noise_vectors_self.neg(), input_vectors_T).sigmoid().log()
        p2_nose_loss_ex = torch.bmm(noise_vectors_output.neg(), self_out_vectors_T).sigmoid().log()
        p1_noise_loss_ex = p1_noise_loss_ex.squeeze().sum(1)
        p2_nose_loss_ex = p2_nose_loss_ex.squeeze().sum(1)

        p1_noise_loss2 = torch.bmm(noise_vectors_input2.neg(), self_in_vectors_T).sigmoid().log()
        p2_nose_loss2 = torch.bmm(noise_vectors_self2.neg(), output_vectors_T).sigmoid().log()
        p1_noise_loss2 = p1_noise_loss2.squeeze().sum(1)
        p2_nose_loss2 = p2_nose_loss2.squeeze().sum(1)

        p1_noise_loss_ex2 = torch.bmm(noise_vectors_self2.neg(), input_vectors_T).sigmoid().log()
        p2_nose_loss_ex2 = torch.bmm(noise_vectors_output2.neg(), self_out_vectors_T).sigmoid().log()
        p1_noise_loss_ex2 = p1_noise_loss_ex2.squeeze().sum(1)
        p2_nose_loss_ex2 = p2_nose_loss_ex2.squeeze().sum(1)

        return -(p1_loss+p2_loss+p1_noise_loss+p2_nose_loss+p1_noise_loss_ex+p2_nose_loss_ex+
                 p1_noise_loss2+p2_nose_loss2+p1_noise_loss_ex2+p2_nose_loss_ex2).mean()
