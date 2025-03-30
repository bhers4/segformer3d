import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.dice_factor = 0.7
        self.bce_factor = 1 - self.dice_factor

    def forward(self, inputs, targets, smooth=1, logits_cast=False):   
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        # BCELoss is unsafe for autocasting for mixed precision
        if logits_cast:
            BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        else:
            BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE*self.bce_factor + dice_loss*self.dice_factor
        
        return Dice_BCE