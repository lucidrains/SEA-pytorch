import pytest
import torch

param = pytest.mark.parametrize

from SEA_pytorch import SEA

def test_sea():
    model = SEA()
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == x.shape
