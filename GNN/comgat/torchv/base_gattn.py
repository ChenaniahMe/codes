import torch
import torch.optim as optim

class BaseGAttN(torch.nn.Module):
    def __init__(self):
        super(BaseGAttN, self).__init__()
        
    # def loss(logits, labels, nb_classes, class_weights):
    #     '''
    #     reference gat from tensorflow, to rewrite torch code.
    #     '''
    #     sample_wts = torch.sum(torch.nn.functional.one_hot(labels, nb_classes)*class_weights,axis=-1)
    #     #代替使用tensorflow，重写代码
    #     y = torch.softmax(logits)
    #     tf_log = torch.log(y)
    #     pixel_wise_mult = labels*tf_log
    #     cross_entropy = -torch.sum(pixel_wise_mult)
    #     xentropy = cross_entropy * sample_wts
    #     return torch.sum(xentropy)

    def masked_softmax_cross_entropy(self,logits, labels, mask):
        # print("logits.shape",logits.shape,"labels.shape",labels.shape,"mask.shape",mask.shape)
        #使用softmax_cross_entropy_with_logits，自己实现
        # print("logits",torch.mean(logits))
        y = torch.softmax(logits,dim=1)
        tf_log = torch.log(y)
        pixel_wise_mult = labels*tf_log
        loss = -pixel_wise_mult
        loss = torch.sum(loss,axis=1)
        # print("torch.sum(loss)",torch.sum(loss))
        # print("mask1",mask)
        # print("loss",loss)
        mask = mask / torch.mean(mask)
        # print("mask2",mask)
        loss = loss * mask
        return torch.mean(loss)

    def masked_accuracy(self,logits, labels, mask):

        accuracy_all = torch.argmax(logits, 1).eq(torch.argmax(labels, 1)).float()
        # print("torch.argmax(logits, 1)",torch.argmax(logits, 1))
        # print("torch.argmax(labels, 1))",torch.argmax(labels, 1))
        # print("accuracy_all",accuracy_all)
        mask /= torch.mean(mask)
        accuracy_all *= mask
        return torch.mean(accuracy_all)
    def training(loss, l2_coef):
        #from torch
        vars = None
        #torch.sum(torch.pow(v,2))/2相当于tensorflow中的l2损失
        lossL2 = torch.add([torch.sum(torch.pow(v,2))/2 for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        
        return 

