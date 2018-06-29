#! /usr/bin/env python
"""
Test fitting.py
"""

from __future__ import print_function

__author__ = 'Paul Hancock'

from AegeanTools import fitting, models
import lmfit
import numpy as np


def make_model():
    """Test that we can make lmfit.Parameter models"""
    model = lmfit.Parameters()
    model.add('c0_amp', 1, vary=True)
    model.add('c0_xo', 5, vary=True)
    model.add('c0_yo', 5, vary=True)
    model.add('c0_sx', 2.001, vary=False)
    model.add('c0_sy', 2, vary=False)
    model.add('c0_theta', 0, vary=False)
    model.add('components', 1, vary=False)
    return model


def test_elliptical_gaussian():
    """Test our elliptical gaussian creation function"""
    x, y = np.indices((3, 3))
    gauss = fitting.elliptical_gaussian(x, y, amp=1, xo=0, yo=1, sx=1, sy=1, theta=0)
    if np.any(np.isnan(gauss)): raise AssertionError()
    gauss = fitting.elliptical_gaussian(x, y, amp=1, xo=0, yo=1, sx=1, sy=1, theta=np.inf)
    if not (np.all(np.isnan(gauss))): raise AssertionError()


def test_CandBmatrix():
    """Test that C and B matricies can be created without error"""
    x, y = map(np.ravel, np.indices((3, 3)))
    C = fitting.Cmatrix(x, y, sx=1, sy=2, theta=0)
    if np.any(np.isnan(C)): raise AssertionError()
    B = fitting.Bmatrix(C)
    if np.any(np.isnan(B)): raise AssertionError()


def test_hessian_shape():
    """Test that the hessian has the correct shape"""
    # test a single component model
    model = make_model()
    nvar = 3
    x, y = np.indices((10, 10))
    Hij = fitting.hessian(model, x, y)
    if not (Hij.shape == (nvar, nvar, 10, 10)): raise AssertionError()

    # now add another component
    model.add('c1_amp', 1, vary=True)
    model.add('c1_xo', 5, vary=True)
    model.add('c1_yo', 5, vary=True)
    model.add('c1_sx', 2.001, vary=True)
    model.add('c1_sy', 2, vary=True)
    model.add('c1_theta', 0, vary=True)
    nvar = 9
    model['components'].value = 2
    Hij = fitting.hessian(model, x, y)
    if not (Hij.shape == (nvar, nvar, 10, 10)): raise AssertionError()


def test_jacobian_shape():
    """
    Test to see if the Jacobian matrix if of the right shape
    This includes a single source model with only 4 variable params
    """
    model = make_model()
    nvar = 3
    x, y = np.indices((10, 10))
    Jij = fitting.jacobian(model, x, y)
    if not (Jij.shape == (nvar, 10, 10)): raise AssertionError()

    model.add('c1_amp', 1, vary=True)
    model.add('c1_xo', 5, vary=True)
    model.add('c1_yo', 5, vary=True)
    model.add('c1_sx', 2.001, vary=True)
    model.add('c1_sy', 2, vary=True)
    model.add('c1_theta', 0, vary=True)
    nvar = 9
    model['components'].value = 2
    Jij = fitting.jacobian(model, x, y)
    if not (Jij.shape == (nvar, 10, 10)): raise AssertionError()


def test_emp_vs_ana_jacobian():
    """Test that the empirical and analytic Jacobians agree"""
    model = make_model()

    x, y = np.indices((10, 10))
    emp_Jij = fitting.emp_jacobian(model, x, y)
    ana_Jij = fitting.jacobian(model, x, y)
    diff = np.abs(ana_Jij - emp_Jij)
    if not (np.max(diff) < 1e-5): raise AssertionError()

    model.add('c1_amp', 1, vary=True)
    model.add('c1_xo', 5, vary=True)
    model.add('c1_yo', 5, vary=True)
    model.add('c1_sx', 2.001, vary=True)
    model.add('c1_sy', 2, vary=True)
    model.add('c1_theta', 0, vary=True)

    model['components'].value = 2
    emp_Jij = fitting.emp_jacobian(model, x, y)
    ana_Jij = fitting.jacobian(model, x, y)
    diff = np.abs(ana_Jij - emp_Jij)
    if not (np.max(diff) < 1e-3): raise AssertionError()


def test_emp_vs_ana_hessian():
    """Test that the empirical and analytical Hessians agree"""
    model = make_model()

    x, y = np.indices((10, 10))
    emp_Hij = fitting.emp_hessian(model, x, y)
    ana_Hij = fitting.hessian(model, x, y)
    diff = np.abs(ana_Hij - emp_Hij)
    if not (np.max(diff) < 1e-5): raise AssertionError()

    model.add('c1_amp', 1, vary=True)
    model.add('c1_xo', 5, vary=True)
    model.add('c1_yo', 5, vary=True)
    model.add('c1_sx', 2.001, vary=True)
    model.add('c1_sy', 2, vary=True)
    model.add('c1_theta', 0, vary=True)

    model['components'].value = 2
    emp_Hij = fitting.emp_hessian(model, x, y)
    ana_Hij = fitting.hessian(model, x, y)
    diff = np.abs(ana_Hij - emp_Hij)
    if not (np.max(diff) < 1): raise AssertionError()


def test_make_ita():
    """Test make_ita"""
    noise = np.random.random((10, 10))
    ita = fitting.make_ita(noise)
    if not (ita.shape == (100, 100)): raise AssertionError()
    noise *= np.nan
    ita = fitting.make_ita(noise)
    if not (len(ita) == 0): raise AssertionError()


def test_RB_bias():
    """Test RB_bias"""
    data = np.random.random((4, 4))
    model = make_model()
    bias = fitting.RB_bias(data, model)
    if not (len(bias) == 3): raise AssertionError()


def test_bias_correct():
    """test that bias_correct does things"""
    data = np.random.random((4, 4))
    model = make_model()
    fitting.bias_correct(model, data)


def test_condon_errs():
    """Test that we can create Condon errors"""
    source = models.OutputSource()
    source.ra = 0
    source.dec = 1
    source.a = 10
    source.b = 10
    source.pa = 0
    source.local_rms = 0.1
    source.peak_flux = 1
    source.int_flux = 1
    fitting.condon_errors(source, None)
    if not (source.err_a == 0): raise AssertionError()
    fitting.condon_errors(source, theta_n=8.)
    if not (source.err_a > 0): raise AssertionError()
    # test that we can get a PA error
    source.a = 20
    fitting.condon_errors(source, None)
    if source.err_pa < 0: raise AssertionError()

if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
