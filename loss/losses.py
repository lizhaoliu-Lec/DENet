import torch
import torch.nn.functional as F


def loss_general(input, target):
    return cross_entropy2d(input.float(), target.long())


def loss_denet(input, target, mode, weight=None):
    if mode == 'knowledge':
        loss = cross_entropy2d(input.float(), target.long(), weight=weight)
    elif mode == 'pattern':
        loss = binary_focal_loss(input.float(), target.long())
    else:
        raise ValueError("mode should be either `knowledge` or `pattern`, but found %s" % mode)
    return loss


def cross_entropy2d(input, target, weight=None, ignore_index=255, reduction='mean'):
    n, c, h, w = input.size()
    if len(list(target.size())) == 3:
        nt, ht, wt = target.size()
    else:
        target = target.squeeze(1)
        nt, ht, wt = target.size()

    # handle inconsistent size between input and target
    # upsample labels
    if h > ht and w > wt:
        target = target.unsqueeze(1)
        target = F.interpolate(target.float(), size=(h, w), mode="nearest").long()
        target = target.squeeze(1)
    # upsample images
    elif h < ht and w < wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.permute(0, 2, 3, 1).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(input, target,
                           weight=weight, ignore_index=ignore_index,
                           reduction=reduction)
    return loss


def binary_focal_loss(input, target, alpha=0.25, gamma=2, ignore_index=255, reduction='mean'):
    weight = torch.tensor([alpha, 1 - alpha])
    if torch.cuda.is_available():
        weight = weight.cuda()
    n, c, h, w = input.size()
    if len(list(target.size())) == 3:
        nt, ht, wt = target.size()
    else:
        target = target.squeeze(1)
        nt, ht, wt = target.size()

    # handle inconsistent size between input and target
    # upsample labels
    if h > ht and w > wt:
        target = target.unsqueeze(1)
        target = F.interpolate(target.float(), size=(h, w), mode="nearest").long()
        target = target.squeeze(1)
    # upsample images
    elif h < ht and w < wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.permute(0, 2, 3, 1).contiguous().view(-1, c)
    target = target.view(-1)
    probs = F.softmax(input, dim=-1)
    log_probs = torch.log(probs)
    logits = (1 - probs) ** gamma * log_probs

    loss = F.nll_loss(logits, target,
                      weight=weight, ignore_index=ignore_index,
                      reduction=reduction)
    return loss
