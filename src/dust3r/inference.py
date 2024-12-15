import torch
import tqdm

from .utils import collate_with_cat, to_cpu


@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f">> Inference with model on {len(pairs)} image pairs")

    result = []

    pairs = [(p[0].to_dict(), p[1].to_dict()) for p in pairs]

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        collated_dict_pairs = collate_with_cat(pairs[i : i + batch_size])
        res = loss_of_one_batch(collated_dict_pairs, model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result)

    return result


def loss_of_one_batch(batch, model, criterion, device, use_amp=False):
    view1, view2 = batch
    ignore_keys = set(["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"])

    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            if criterion is None:
                loss = None
            loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    return dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
