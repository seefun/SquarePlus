import torch
from squareplus import SquarePlus


class TestModel:
    feature = torch.rand(2, 32, 56, 56)

    def test_squareplus(self):
        act = SquarePlus()
        assert act(TestModel.feature).shape == (2, 32, 56, 56)
