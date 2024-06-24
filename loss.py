from typing import Optional, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(loss_function: str, class_weights: Optional[Union[np.ndarray, float]], focal_loss_alpha: float = 0.25, focal_loss_gamma: float = 2.0, gradient_penalty: float = 1.):
    if loss_function == "FL":
        criterion = BinaryFocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
    elif loss_function == "CE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == "WCE":
        criterion = WeightedBCEWithLogitsLoss(class_weights)
    elif loss_function == "MSE":
        criterion = SigmoidMSELoss()
    elif loss_function == "L1":
        criterion = SigmoidL1Loss()
    elif loss_function == "SL1":
        criterion = SigmoidSmoothL1Loss()
    elif loss_function == "GLiP":
        criterion = GLiPLoss(gradient_penalty)
    elif loss_function == "NGLiP":
        criterion = NormGLiPLoss(gradient_penalty)
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")
    return criterion


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight: float = 0.5, epsilon: float = 1e-7):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        assert weight >= 0 and weight <= 1
        self.w_p = 2 * weight
        self.w_n = 2 - self.w_p
        self.epsilon = epsilon
        
    def forward(self, logits, labels):
        ps = torch.sigmoid(logits.squeeze()) 

        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + self.epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + self.epsilon))

        loss = loss_pos + loss_neg
        
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self, weight: float = 0.5, epsilon: float = 1e-7):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        assert weight >= 0 and weight <= 1
        self.w_p = 2 * weight
        self.w_n = 2 - self.w_p
        self.epsilon = epsilon
        
    def forward(self, ps, labels):
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + self.epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + self.epsilon))

        loss = loss_pos + loss_neg
        
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: int = 2, weight: Optional[torch.Tensor] = None, size_average: bool = True, reduction: Optional[str] = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):        
        # first compute binary cross-entropy 
        CE = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        CE_EXP = torch.exp(-CE)
        focal_loss = self.alpha * (1-CE_EXP)**self.gamma * CE
        
        # reduce loss
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: int = 2, weight: Optional[torch.Tensor] = None, size_average: bool = True, reduction: Optional[str] = 'mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction="none")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
        
        # reduce loss
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss



class DiceLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, size_average: bool = True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, size_average: bool = True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


class SigmoidMSELoss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean'):
        super(SigmoidMSELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        loss = F.mse_loss(inputs, targets, reduction='none')

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss


class SigmoidL1Loss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean'):
        super(SigmoidL1Loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        loss = F.l1_loss(inputs, targets, reduction='none')

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss
    

class SigmoidSmoothL1Loss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean', beta: float = 1.0):
        super(SigmoidSmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.beta = beta
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        loss = F.smooth_l1_loss(inputs, targets, reduction='none', beta=self.beta)

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss



class SoftmaxMSELoss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean'):
        super(SoftmaxMSELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        loss = F.mse_loss(inputs, targets, reduction='none')

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss


class SoftmaxL1Loss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean'):
        super(SoftmaxL1Loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        loss = F.l1_loss(inputs, targets, reduction='none')

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss

class SoftmaxSmoothL1Loss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean', beta: float = 1.0):
        super(SoftmaxSmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.beta = beta
        
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        loss = F.smooth_l1_loss(inputs, targets, reduction='none', beta=self.beta)

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss


class NormGLiPLoss(nn.Module):
    def __init__(self, gradient_penalty: float = 1., reduction: Optional[str] = 'mean'):
        super(NormGLiPLoss, self).__init__()
        self.gradient_penalty = gradient_penalty
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size,3:n_channels,W,H,D)
            targets: (batch_size,3:n_channels,W,H,D)
        """
        shapes = inputs.shape[2:]

        # compute gradient penalty
        center_difference = []
        for i, s_i in enumerate(shapes):
            f = inputs
            for j, s_j in enumerate(shapes):
                if i!=j:
                    f = torch.narrow(f, j+2, 1, s_j-2)
            center_difference.append(torch.narrow(f, i+2, 0, s_i-2) + torch.narrow(f, i+2, 2, s_i-2) - 2 * torch.narrow(f, i+2, 1, s_i-2))
        center_difference = torch.stack(center_difference, dim=-1) #(bs,nc,W,H,D,xyz)
        gradient_penalty = self.gradient_penalty * (torch.linalg.norm(center_difference, dim=-1) - 1).pow(2).mean([i+2 for i in range(len(shapes))]) #(bs,nc)

        # compute wasserstein loss
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1) #(bs,nc,W*H*D)
        targets = targets.reshape(targets.shape[0], targets.shape[1], -1) #(bs,nc,W*H*D)
        wasserstein_loss = - (inputs * targets).sum(2) / targets.sum([0,2]) \
            + (inputs * (1 - targets)).sum(2) / (1 - targets).sum([0,2]) #(bs,nc)

        # compute loss
        loss = wasserstein_loss + gradient_penalty
        
        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class GLiPLoss(nn.Module):
    def __init__(self, gradient_penalty: float = 1., reduction: Optional[str] = 'mean'):
        super(GLiPLoss, self).__init__()
        self.gradient_penalty = gradient_penalty
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size,3:n_channels,W,H,D)
            targets: (batch_size,3:n_channels,W,H,D)
        """
        shapes = inputs.shape[2:]

        # compute gradient penalty
        gradient_penalty = 0
        for i, s in enumerate(shapes):
            gradient_penalty += self.gradient_penalty * (torch.abs(torch.narrow(inputs, i+2, 1, s-1) - torch.narrow(inputs, i+2, 0, s-1)) - 1/math.sqrt(len(shapes))).pow(2).mean([j+2 for j in range(len(shapes))]) #(bs,nc)

        # compute wasserstein loss
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1) #(bs,nc,W*H*D)
        targets = targets.reshape(targets.shape[0], targets.shape[1], -1) #(bs,nc,W*H*D)
        wasserstein_loss = - (inputs * targets).sum(2) / targets.sum([0,2]) \
            + (inputs * (1 - targets)).sum(2) / (1 - targets).sum([0,2]) #(bs,nc)

        # compute loss
        loss = wasserstein_loss + gradient_penalty
        
        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class ImageLipschitzLoss(nn.Module):
    def __init__(self, penalty_coefficient: float = 1., lipschitz_coefficient: float = 1, reduction: Optional[str] = 'mean'):
        super(ImageLipschitzLoss, self).__init__()
        self.penalty_coefficient = penalty_coefficient
        self.lipschitz_coefficient = lipschitz_coefficient
        self.reduction = reduction

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs: (batch_size,n_channels,W,H,D)
        """
        shapes = inputs.shape[2:]

        # compute gradient penalty
        loss = 0
        for i, s in enumerate(shapes):
            loss += self.penalty_coefficient * (torch.abs(torch.narrow(inputs, i+2, 1, s-1) - torch.narrow(inputs, i+2, 0, s-1)) - self.lipschitz_coefficient/math.sqrt(len(shapes))).pow(2).mean([j+2 for j in range(len(shapes))]) #(bs,nc)

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class ImageGradientConsistentLoss(nn.Module):
    def __init__(self, penalty_coefficient: float = 1., power: int = 2, reduction: Optional[str] = 'mean'):
        super(ImageGradientConsistentLoss, self).__init__()
        self.penalty_coefficient = penalty_coefficient
        self.power = power
        self.reduction = reduction

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs: (batch_size,n_channels,W,H,D)
        """
        shapes = inputs.shape[2:]

        # compute gradient penalty
        loss = 0
        for i, s in enumerate(shapes):
            gradients = torch.narrow(inputs, i+2, 1, s-1) - torch.narrow(inputs, i+2, 0, s-1) #(bs,nc,W-1,H,D)
            double_gradient = torch.narrow(gradients, i+2, 1, s-2) - torch.narrow(gradients, i+2, 0, s-2) #(bs,nc,W-2,H,D)
            loss += torch.abs(double_gradient).pow(self.power).mean([j+2 for j in range(len(shapes))]) #(bs,nc)

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss