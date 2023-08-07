import pytest
import pink

def test_pink():
    pink.Interpolation
    assert pink.doc() == "PINK python interface"
    assert pink.__version__ == "2.5"
