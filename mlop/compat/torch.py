import logging

import torch

import mlop

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Torch"


def watch(
    module: torch.nn.Module,
    disable_graph: bool = False,
    disable_grad: bool = False,
    disable_param: bool = False,
    freq: int | None = 1000,
    bins: int | None = 64,
    **kwargs,
):
    # TODO: remove legacy compat
    if "log" in kwargs:
        disable_grad = kwargs["log"] not in ["gradients", "all"]
        disable_param = kwargs["log"] not in ["parameters", "all"]
    if "log_freq" in kwargs:
        freq = kwargs["log_freq"]
    if "log_graph" in kwargs:
        disable_graph = not kwargs["log_graph"]

    if mlop.ops is None or len(mlop.ops) == 0:
        logger.critical(f"{tag}: no runs to attach, please call mlop.init() first")
        return
    else:
        op, hooks = mlop.ops[-1], mlop._hooks
        op._torch, module._nodes = module, {}

    if not disable_grad:
        for name, param in module.named_parameters():
            if param.requires_grad and check_param(param, name):
                hooks.append(param.register_hook(_backward(op, name, freq, bins)))

    if not disable_param:
        hooks.append(module.register_forward_hook(_forward(op, freq, bins)))

    if not disable_graph:
        types = [
            getattr(torch.nn, t)
            for t in (
                "Container",
                "Sequential",
                "ModuleList",
                "ModuleDict",
            )
            if hasattr(torch.nn, t)
        ]
        op._torch._nodes = _to_dict(module)
        hooks.append(module.register_forward_hook(_forward_child(op)))
        hooks.extend(_hook_child(op, types, module, hooks))
    return hooks


def _to_dict(module):
    d = get_module(module)

    children = []
    for idx, (name, child) in enumerate(module._modules.items()):
        if child is None:
            continue
        child_dict = {"name": name, "order": idx}
        child_dict.update(_to_dict(child))
        children.append(child_dict)

    if children:
        d["children"] = children

    return d


def _hook_child(op, types, module, hooks=[], prefix=None):
    for name, child in module.named_children():
        if prefix:
            name = f"{prefix}.{name}"
        try:
            hooks.append(child.register_forward_hook(_forward_child(op)))
            hooks = _hook_child(op, types, child, hooks, name)
        except RuntimeError as e:
            logger.error("%s: failed to watch child %s: %s", tag, name, e)
    return hooks


def _forward_child(op):
    c = [0]

    def f(module, input, output):
        if c[0] == 1:
            if op._torch._nodes:
                if not hasattr(op._torch, "_ready"):
                    op._torch._ready = True
            return
        c[0] = 1

        if not isinstance(input, tuple):
            input = (input,)
        if not isinstance(output, tuple):
            output = (output,)

        n = _find_node(op._torch._nodes, id(module))
        if n is not None:
            n.update(
                {
                    "input": {
                        "id": [
                            t.data_ptr() if hasattr(t, "data_ptr") else id(t)
                            for t in input
                        ],
                        "shape": [get_shape(i) for i in input],
                        "grad_fn": [
                            id(i.grad_fn) if hasattr(i, "grad_fn") else None
                            for i in input
                        ],
                        "next": [
                            [id(f) for f, _ in i.grad_fn.next_functions]
                            if hasattr(i.grad_fn, "next_functions")
                            else None
                            for i in input
                        ],
                    },
                    "output": {
                        "id": [
                            t.data_ptr() if hasattr(t, "data_ptr") else id(t)
                            for t in output
                        ],
                        "shape": [get_shape(i) for i in output],
                        "grad_fn": [
                            id(i.grad_fn) if hasattr(i, "grad_fn") else None
                            for i in output
                        ],
                        "next": [
                            [id(f) for f, _ in i.grad_fn.next_functions]
                            if hasattr(i.grad_fn, "next_functions")
                            else None
                            for i in output
                        ],
                    },
                }
            )

    return f


def _backward(op, name, freq, bins):
    c = [0]

    def f(grad):
        c[0] += 1
        if c[0] < freq:
            return
        c[0] = 0
        hist = make_compat_histogram_tensor(grad.data, bins)
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
                hist = make_compat_histogram_tensor(param.data, bins)
                if hist is not None:
                    op.log({f"{op.settings.x_param_label}/{name}": hist}, step=op._step)
                else:
                    logger.error(f"{tag}: {name} does not contain a valid tensor")

    return f


def check_param(param, name):
    if isinstance(param, torch.autograd.Variable):
        return True
    else:
        logger.error(
            f"{tag}: {name} is of type {type(param).__module__}.{type(param).__name__} and not a torch.Variable"
        )
        return False


def get_module(module):
    info = {
        "type": module.__class__.__name__,
        "id": id(module),
        "args": [],
        "kwargs": {},
    }

    if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
        info["args"] = [module.in_channels, module.out_channels]
    elif hasattr(module, "in_features") and hasattr(module, "out_features"):
        info["args"] = [module.in_features, module.out_features]

    for k, v in module.__dict__.items():  # dir(module)
        if not k.startswith("_") and not callable(v):  # skip private attrs and methods
            if isinstance(v, torch.Size):
                v = tuple(v)
            elif hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    continue
            if (
                v is not None
                and v != ()
                and v != []
                and isinstance(v, (int, float, str, bool))
            ):
                info["kwargs"][k] = v

    # TODO: find a more robust solution to enforce params
    params = {
        pname: list(param.size())
        for pname, param in module.named_parameters()
        if "." not in pname
    }
    if params:
        info.update({"params": params})

    return info


def get_shape(tensor, r=set()):
    if hasattr(tensor, "size"):
        return list(tensor.size())  # pytorch
    elif hasattr(tensor, "get_shape"):
        return tensor.get_shape().as_list()  # tensorflow
    elif hasattr(tensor, "shape"):
        return tensor.shape

    try:
        r.add(id(tensor))
        return [get_shape(i, r) if id(i) not in r else 0 for i in tensor]
    except TypeError:
        logger.error(f"{tag}: {tensor} is not iterable")
        return []


def make_compat_histogram_tensor(tensor, bins=64):
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


def _find_node(nodes, target_id):
    if nodes.get("id") == target_id:
        return nodes

    if "children" in nodes:
        for child in nodes["children"]:
            result = _find_node(child, target_id)
            if result is not None:
                return result

    return None
