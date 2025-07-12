import torch

def run_optimizer_step(loss, optimizer, model, max_norm, return_grad_norm=False):
    optimizer.zero_grad()
    loss.backward()

    # Pre-clipping
    pre_clip_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            pre_clip_norm += p.grad.data.norm().item() ** 2
    pre_clip_norm = pre_clip_norm ** 0.5

    # Clip and step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()

    # Post-clipping
    post_clip_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            post_clip_norm += p.grad.data.norm().item() ** 2
    post_clip_norm = post_clip_norm ** 0.5

    if return_grad_norm:
        return pre_clip_norm, post_clip_norm