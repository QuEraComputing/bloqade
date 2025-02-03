import zlib
import base64
from typing import Any, Dict

import numpy as np


def bytes_to_str(byte_array: bytes) -> str:
    compressed_bytes = zlib.compress(byte_array)
    return base64.b64encode(compressed_bytes).decode("ascii")


def str_to_bytes(string: str) -> bytes:
    compressed_bytes = base64.b64decode(string)
    return zlib.decompress(compressed_bytes)


def validate_dtype(dtype) -> None:
    if np.dtype(dtype) == np.dtype(object):
        raise ValueError("Object arrays are not supported")
    elif dtype.subdtype is not None:
        validate_dtype(dtype.subdtype[0])
    elif dtype.fields is not None:
        for _, (field_dtype, _) in dtype.fields.items():
            validate_dtype(field_dtype)


def array_to_json(arr: np.ndarray) -> Dict[str, Any]:
    validate_dtype(arr.dtype)

    return {
        "dtype": (arr.dtype.descr if arr.dtype.fields is not None else arr.dtype.str),
        "shape": arr.shape,
        "buffer": bytes_to_str(arr.tobytes()),
    }


def json_to_array(json_dict: dict) -> np.ndarray:
    _data = json_dict["buffer"]
    _dtype = json_dict["dtype"]
    _shape = json_dict["shape"]

    return np.frombuffer(str_to_bytes(_data), dtype=_dtype).reshape(_shape)
