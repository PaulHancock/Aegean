from AegeanTools.BANE_fft import robust_bane, pad
import numpy as np

rng = np.random.default_rng(12345)

def test_pad_constant():
    """
    Test the pad function.
    """

    array = rng.random((100, 100))
    array_pad = pad(
        array=array,
        pad_width=(10, 10),
        mode='constant',
        constant_values=0,
    )
    array_pad_np = np.pad(
        array=array,
        pad_width=(10, 10),
        mode='constant',
        constant_values=0,
    )
    assert np.array_equal(array_pad, array_pad_np), "Pad function with constant mode failed"
    assert array_pad.shape == (120, 120), f"Bad shape {array_pad.shape} != (120, 120)"
    assert np.all(array_pad[0, :] == 0), "Padding not constant"

def test_pad_reflect():
    """
    Test the pad function.
    """

    array = rng.random((100, 100))
    array_pad = pad(
        array=array,
        pad_width=(10, 10),
        mode='reflect',
    )
    array_pad_np = np.pad(
        array=array,
        pad_width=(10, 10),
        mode='reflect',
    )
    assert np.array_equal(array_pad, array_pad_np), "Pad function with reflect mode failed"
    assert array_pad.shape == (120, 120), f"Bad shape {array_pad.shape} != (120, 120)"