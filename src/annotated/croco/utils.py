def to_2tuple(x):
    """Convert input to a tuple of length 2.

    This is used to allow flexible specification of per-layer parameters.
    For example, if you want different dropout rates for each layer,
    you can pass a tuple like (0.1, 0.2). If you want the same value
    for both layers, you can pass a single value like 0.1 which will
    be converted to (0.1, 0.1).

    Args:
        x: Either a single value or a tuple of two values

    Returns:
        A tuple of two values, either the input tuple or the input value repeated
    """
    if isinstance(x, tuple):
        return x

    if isinstance(x, list):
        return tuple(x)

    return (x, x)
