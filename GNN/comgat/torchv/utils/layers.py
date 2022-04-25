import numpy as np
import torch

def attn_head(seqf,seqf1,seqf2,bias,seq,indrop,coefdrop, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    # print("seq.shape",seq.shape)
    # if in_drop != 0.0:
    #     # seq = torch.nn.Dropout(1.0 - in_drop)(seq)
    #     seq = indrop(seq)
    seq_fts = seqf(seq)
    f_1 = seqf1(seq_fts)
    f_2 = seqf2(seq_fts)
    # print("f_1.shape",f_1.shape)
    # print("f_2.shape",f_2.shape)
    logits = f_1 + torch.transpose(f_2, 2,1)
    # print("logits.shape",logits.shape)
    # print("logits.shape",logits.shape)
    rel = torch.nn.LeakyReLU()(logits)
    # rel = logits
    # print("rel.shape",rel.shape)
    coefs = torch.softmax(rel+bias_mat,dim=1)
    # # print(coefs)
    # if coef_drop != 0.0:
    #     coefs = coefdrop(coefs)
    # if in_drop != 0.0:
    #     seq_fts = indrop(seq_fts)

    vals = coefs@seq_fts
    # print("bias",bias)
    # vals = vals + bias.cuda()
    # from torch.autograd import gradcheck
    # print("bias.cuda()",bias.cuda())
    # ret = 
    # print("vals.shape",vals.shape)
    return activation(vals)