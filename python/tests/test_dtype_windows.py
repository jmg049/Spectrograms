"""dtype coverage for the WindowType.make_* window generators."""

import numpy as np
import pytest
import spectrograms as sg

WINDOW_MAKERS = [
    ("make_hanning", (512,)),
    ("make_hamming", (512,)),
    ("make_blackman", (512,)),
    ("make_kaiser", (512, 8.0)),
    ("make_gaussian", (512, 0.4)),
]


@pytest.mark.parametrize("maker,args", WINDOW_MAKERS)
def test_window_default_is_float64(maker, args):
    w = getattr(sg.WindowType, maker)(*args)
    assert w.dtype == np.float64
    assert w.shape == (512,)


@pytest.mark.parametrize("maker,args", WINDOW_MAKERS)
def test_window_float32(maker, args):
    fn = getattr(sg.WindowType, maker)
    w64 = fn(*args, dtype="float64")
    w32 = fn(*args, dtype="float32")
    assert w64.dtype == np.float64
    assert w32.dtype == np.float32
    assert w32.shape == (512,)
    assert np.allclose(w32, w64, atol=1e-6)


def test_window_invalid_dtype():
    with pytest.raises(ValueError):
        sg.WindowType.make_hanning(512, dtype="float16")
