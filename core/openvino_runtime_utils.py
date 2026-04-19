def get_openvino_input_name(model_obj) -> str:
    try:
        port = model_obj.input(0)
    except Exception:
        port = model_obj.inputs[0]

    try:
        name = port.get_any_name()
        if name:
            return name
    except Exception:
        pass

    try:
        names = list(port.get_names())
        if names:
            return names[0]
    except Exception:
        pass

    return "images"


def resolve_square_input_size(compiled_model, fallback_size: int) -> int:
    try:
        shape = list(compiled_model.input(0).shape)
    except Exception:
        return fallback_size

    if (
        len(shape) == 4
        and int(shape[2]) > 0
        and int(shape[3]) > 0
        and int(shape[2]) == int(shape[3])
    ):
        return int(shape[2])
    return fallback_size
