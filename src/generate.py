import torch

from src import config


def generate_and_decode_next_patch(model, input_sequence, device, patch_size):
    """
    Given an input sequence, generate the next patch and decode it into spatiotemporal points.
    """
    model.eval()
    input_sequence = input_sequence.to(device)
    with torch.no_grad():
        outputs = model(input_sequence)
        # The last output token predicts the next patch
        next_patch_flat = outputs[:, -1, :]
    return next_patch_flat


