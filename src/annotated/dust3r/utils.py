import torch


def fill_default_args(kwargs, func):
    import inspect  # a bit hacky but it works reliably

    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def is_symmetrized(gt1, gt2):
    x = gt1["instance"]
    y = gt2["instance"]
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def transpose_to_landscape(head, activate=True):
    """Predict in the correct aspect-ratio,
    then transpose the result in landscape
    and stack everything back together.
    """

    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), "true_shape must be all identical"
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = width >= height
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))

        # batch is a mix of both portraint & landscape
        def selout(ar):
            return [d[ar] for d in decout]

        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait), (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}
