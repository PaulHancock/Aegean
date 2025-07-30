#! /usr/bin/env python
"""
Test fitting.py
"""
import lmfit
import numpy as np
from AegeanTools import fitting

__author__ = "Paul Hancock"


def make_model():
    """Test that we can make lmfit.Parameter models"""
    model = lmfit.Parameters()
    model.add("c0_amp", 1, vary=True)
    model.add("c0_xo", 5, vary=True)
    model.add("c0_yo", 5, vary=True)
    model.add("c0_sx", 2.001, vary=False)
    model.add("c0_sy", 2, vary=False)
    model.add("c0_theta", 0, vary=False)
    model.add("components", 1, vary=False)
    return model


def test_elliptical_gaussian():
    """Test our elliptical gaussian creation function"""
    x, y = np.indices((3, 3))
    gauss = fitting.elliptical_gaussian(x, y, amp=1, xo=0, yo=1, sx=1, sy=1, theta=0)
    if np.any(np.isnan(gauss)):
        raise AssertionError()
    gauss = fitting.elliptical_gaussian(
        x, y, amp=1, xo=0, yo=1, sx=1, sy=1, theta=np.inf
    )
    if not (np.all(np.isnan(gauss))):
        raise AssertionError()


def test_CandBmatrix():
    """Test that C and B matricies can be created without error"""
    x, y = map(np.ravel, np.indices((3, 3)))
    C = fitting.Cmatrix(x, y, sx=1, sy=2, theta=0)
    if np.any(np.isnan(C)):
        raise AssertionError()
    B = fitting.Bmatrix(C)
    if np.any(np.isnan(B)):
        raise AssertionError()


def test_jacobian_shape():
    """
    Test to see if the Jacobian matrix if of the right shape
    This includes a single source model with only 4 variable params
    """
    model = make_model()
    nvar = 3
    x, y = np.indices((10, 10))
    Jij = fitting.jacobian(model, x, y)
    if not (Jij.shape == (nvar, 10, 10)):
        raise AssertionError()

    model.add("c1_amp", 1, vary=True)
    model.add("c1_xo", 5, vary=True)
    model.add("c1_yo", 5, vary=True)
    model.add("c1_sx", 2.001, vary=True)
    model.add("c1_sy", 2, vary=True)
    model.add("c1_theta", 0, vary=True)
    nvar = 9
    model["components"].value = 2
    Jij = fitting.jacobian(model, x, y)
    if not (Jij.shape == (nvar, 10, 10)):
        raise AssertionError()


def test_emp_vs_ana_jacobian():
    """Test that the empirical and analytic Jacobians agree"""
    model = make_model()

    x, y = np.indices((10, 10))
    emp_Jij = fitting.emp_jacobian(model, x, y)
    ana_Jij = fitting.jacobian(model, x, y)
    diff = np.abs(ana_Jij - emp_Jij)
    if not (np.max(diff) < 1e-5):
        raise AssertionError()

    model.add("c1_amp", 1, vary=True)
    model.add("c1_xo", 5, vary=True)
    model.add("c1_yo", 5, vary=True)
    model.add("c1_sx", 2.001, vary=True)
    model.add("c1_sy", 2, vary=True)
    model.add("c1_theta", 0, vary=True)

    model["components"].value = 2
    emp_Jij = fitting.emp_jacobian(model, x, y)
    ana_Jij = fitting.jacobian(model, x, y)
    diff = np.abs(ana_Jij - emp_Jij)
    if not (np.max(diff) < 1e-3):
        raise AssertionError()


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
