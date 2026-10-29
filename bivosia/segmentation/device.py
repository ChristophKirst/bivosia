"""Select an available PyTorch execution device."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'


def resolve_device(device):
    """Choose working CUDA in auto mode, otherwise CPU; respect explicit choices."""
    import torch

    if device == 'cpu':
        return device
    if device not in ('auto', 'cuda', 'mps') and not device.startswith('cuda:'):
        raise ValueError('Choose auto, cuda, cuda:N, mps or cpu')
    selected = 'cuda' if device == 'auto' else device
    try:
        torch.empty(1, device=selected)
        return selected
    except Exception:
        if device != 'auto':
            raise RuntimeError(f'Device is unavailable: {device}') from None
        return 'cpu'
