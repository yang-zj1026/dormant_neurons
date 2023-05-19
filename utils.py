import torch


def tensor_tree_map(fn, tree):
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    elif isinstance(tree, dict):
        return {k: tensor_tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tensor_tree_map(fn, item) for item in tree)
    else:
        raise TypeError(f"Unsupported type: {type(tree)}")


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(flat_dict, sep='/'):
    unflattened_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(sep)
        current_dict = unflattened_dict
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    return unflattened_dict


def split_key(key, num=2):
    """Split the random key into multiple keys using PyTorch.

    Args:
        key: Random key as a string.
        num: Number of keys to split into.

    Returns:
        A list of `num` torch.Tensor keys.
    """
    torch.manual_seed(int(key))
    keys = [torch.randn(()) for _ in range(num)]
    return keys
