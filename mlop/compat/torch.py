import logging

import torch

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Torch"


def watch(
    module: torch.nn.Module,
    log: str | None = "gradients",
    log_freq: int | None = 1000,
):
    if not hasattr(module, "_watched"):
        module._watched = []

    c = [0, log_freq]

    for name, param in module.named_parameters():
        if param.requires_grad:
            module._watched.append(log)

        if not isinstance(param, torch.autograd.Variable):
            logger.critical(
                f"{tag}: {name} is of type {type(param).__module__}.{type(param).__name__} and not a torch.Variable"
            )
            continue

        # TODO: keep track of hook handles
        def _callback(grad, c=[0, log_freq]):
            c[0] += 1
            if c[0] < c[1]:
                return
            c[0] = 0
            tensor = grad.data
            print(tensor)  # op.log on the tensor (log_tensor_stats)

        handle = param.register_hook(lambda grad: _callback(grad, c))
        return handle
