


def masked_l1_loss_batches(preds, target, mask_valid):
    '''used for winrate test, compute the loss for each image'''
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0

    losses = element_wise_loss.sum() / mask_valid.sum()
    return losses, element_wise_loss.mean(1), element_wise_loss.mean((1,2,3))
    

def masked_l1_loss(preds, target, mask_valid):
    assert preds.size() == target.size()

    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()


def masked_mse_loss(preds, target, mask_valid):
    element_wise_loss = (preds - target)**2
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()


def compute_grad_norm_losses(losses, model):
    '''
    Balances multiple losses by weighting them inversly proportional
    to their overall gradient contribution.
    
    Args:
        losses: A dictionary of losses.
        model: A PyTorch model.
    Returns:
        A dictionary of loss weights.
    '''
    grad_norms = {}
    model.zero_grad()
    for loss_name, loss in losses.items():
        loss.backward(retain_graph=True)
        grad_sum = sum([w.grad.abs().sum().item() for w in model.parameters() if w.grad is not None])
        num_elem = sum([w.numel() for w in model.parameters() if w.grad is not None])
        grad_norms[loss_name] = grad_sum / num_elem
        model.zero_grad()

    grad_norms_total = sum(grad_norms.values())

    loss_weights = {}
    for loss_name, loss in losses.items():
        weight = (grad_norms_total - grad_norms[loss_name]) / ((len(losses) - 1) * grad_norms_total)
        loss_weights[loss_name] = weight
        
    return loss_weights