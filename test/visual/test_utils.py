from bloqade.visual.animation.runtime import utils
import json
import numpy as np
import pytest


def test_json_numpy():
    original_array = np.random.uniform(size=(1, 5, 3, 2, 4))
    array_json = utils.array_to_json(original_array)
    # serialize then deserialize to ensure the json is in the correct format
    obj = json.loads(json.dumps(array_json))
    # unpack the json object
    unpacked_array = utils.json_to_array(obj)
    # check that the array is the same bytes
    assert unpacked_array.tobytes() == original_array.tobytes()


def test_validation():
    a = np.array([((1, 2), 2), ((3, 4), 4)], dtype=[("x", "O", (2,)), ("y", "i8")])

    with pytest.raises(ValueError):
        utils.array_to_json(a)
