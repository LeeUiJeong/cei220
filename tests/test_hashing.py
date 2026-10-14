from collections import OrderedDict

import torch

from cei220.hashing import hash_state_dict


def test_state_hash_is_order_independent_by_key():
    a = OrderedDict([("b", torch.tensor([2.0])), ("a", torch.tensor([1.0]))])
    b = OrderedDict([("a", torch.tensor([1.0])), ("b", torch.tensor([2.0]))])
    assert hash_state_dict(a) == hash_state_dict(b)


def test_state_hash_changes_with_content():
    a = {"a": torch.tensor([1.0])}
    b = {"a": torch.tensor([2.0])}
    assert hash_state_dict(a) != hash_state_dict(b)
