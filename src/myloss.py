from torch import nn 
import torch
class MSELoss(nn.Module):
    def __init__(self, loss='missing'):
        super(MSELoss, self).__init__()
        #self.args = args
        self.loss = loss
        assert self.loss in ['missing', 'init', 'both']
        self.mseloss = nn.MSELoss()

    def forward(self, pred, label, mask, visit_mask=None):

            # visit_mask: shape [B, T]
        if visit_mask is not None:
            valid_mask = (mask == 1) & visit_mask.bool().unsqueeze(-1)
        else:
            valid_mask = (mask == 1)

        if valid_mask.sum() == 0:
            print("⚠️ No valid missing values to compute loss")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        #print(pred.shape)
        #print(mask.shape) # #torch.Size([2, 1826, 15])
        #print(label.shape)
        pred = pred.view(-1)
        label = label.view(-1)
        mask = valid_mask.view(-1)
        assert len(pred) == len(label) == len(mask)

        indices = mask == 1
        ipred = pred[indices]
        ilabel = label[indices]
        loss = self.mseloss(ipred, ilabel)

        if (mask == 1).sum() == 0:
            print("No missing values to evaluate in this batch")

        if self.loss == 'both':
            non_masked = (mask == 0)
            indices = non_masked
            ipred = pred[indices]
            ilabel = label[indices]
            loss += self.mseloss(ipred, ilabel) # * 0.1

        #print('pred.shape', pred.size()) #  torch.Size([54780])
        return loss

