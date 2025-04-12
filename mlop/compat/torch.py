import logging

import torch

import mlop

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Torch"


def watch(
    module: torch.nn.Module,
    # log: str | None = "gradients",
    disable_grad: bool = False,
    disable_param: bool = False,
    freq: int | None = 1000,  # log_freq
    bins: int | None = 64,
):
    if mlop.ops is None or len(mlop.ops) == 0:
        logger.critical(f"{tag}: no runs to attach, please call mlop.init() first")
        return
    else:
        op = mlop.ops[-1]
        log, hooks = op.log, mlop._hooks

    if not disable_grad:
        for name, param in module.named_parameters():
            if param.requires_grad and check_param(param, name):
                hooks.append(
                    param.register_hook(_backward(op, name, freq, bins))
                )

    if not disable_param:
        hooks.append(module.register_forward_hook(_forward(op, freq, bins)))

    return hooks


def check_param(param, name):
    if isinstance(param, torch.autograd.Variable):
        return True
    else:
        logger.error(
            f"{tag}: {name} is of type {type(param).__module__}.{type(param).__name__} and not a torch.Variable"
        )
        return False


def _backward(op, name, freq, bins):
    c = [0]

    def f(grad):
        c[0] += 1
        if c[0] < freq:
            return
        c[0] = 0
        hist = make_compat_histogram_torch(grad.data, bins)
        if hist is not None:
            op.log({f"{op.settings.x_grad_label}/{name}": hist}, step=op._step)

    return f


def _forward(op, freq, bins):
    c = [0]

    def f(module, input, output):
        c[0] += 1
        if c[0] < freq:
            return
        c[0] = 0

        for name, param in module.named_parameters():
            if check_param(param, name):
                hist = make_compat_histogram_torch(param.data, bins)
                if hist is not None:
                    op.log({f"{op.settings.x_param_label}/{name}": hist}, step=op._step)
                else:
                    logger.error(f"{tag}: {name} does not contain a valid tensor")

    return f


def make_compat_histogram_torch(tensor, bins=64):
    if isinstance(tensor, (tuple, list)):
        tensor = torch.cat([t.detach().clone().reshape(-1) for t in tensor])
    tensor = tensor.detach().clone()

    # handle sparse tensor zeros
    zeros = None
    if tensor.is_sparse:
        tensor = tensor.cpu().coalesce()
        values = tensor._values()
        zeros = tensor.numel() - values.numel()
        tensor = values

    flat = tensor.reshape(-1)
    if flat.is_cuda:
        try:
            flat.histc(bins=64)  # check for histc support
            if not isinstance(flat, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
                flat = flat.type(torch.cuda.FloatTensor)
        except RuntimeError:
            flat = flat.cpu()
    if not flat.is_cuda and not isinstance(
        flat, (torch.FloatTensor, torch.DoubleTensor)
    ):
        flat = flat.type(torch.FloatTensor)

    flat = make_compat_tensor(flat)
    if flat is None:
        return None

    # find histogram bounds
    tmin, tmax = flat.min().item(), flat.max().item()
    if zeros:
        tmin = min(0, tmin)
        tmax = max(0, tmax)
    if tmin > tmax:
        tmin, tmax = tmax, tmin

    if True:  # tmin != tmax:
        tensor = flat.histc(bins=bins, min=tmin, max=tmax)
        bins = torch.linspace(tmin, tmax, steps=bins + 1)
    else:  # use single bin if all values are the same
        tensor = torch.Tensor([flat.numel()])
        bins = torch.Tensor([tmin, tmax])

    # add back zeros from sparse tensor
    if zeros:
        mask = (bins[:-1] <= 0) & (bins[1:] > 0)
        if not mask.any():
            mask = torch.zeros_like(bins[:-1], dtype=torch.bool)
            mask[-1] = bins[-1] == 0
        tensor[mask] += zeros

    return mlop.Histogram(data=(tensor.tolist(), bins.tolist()), bins=None)


def make_compat_tensor(tensor):
    if tensor.shape == torch.Size([0]) or (~torch.isfinite(tensor)).all().item():
        return None  # invalid if empty or all inf/nan
    elif not torch.isfinite(tensor).all():
        return tensor[torch.isfinite(tensor)]  # remove inf/nan
    else:
        return tensor
