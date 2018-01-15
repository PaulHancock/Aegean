#! python
from __future__ import print_function

__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import fitting, models
import lmfit
import numpy as np


def make_model():
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
    x, y = np.indices((3, 3))
    gauss = fitting.elliptical_gaussian(x, y, amp=1, xo=0, yo=1, sx=1, sy=1, theta=0)
    assert not np.any(np.isnan(gauss))
    gauss = fitting.elliptical_gaussian(x, y, amp=1, xo=0, yo=1, sx=1, sy=1, theta=np.inf)
    assert np.all(np.isnan(gauss))


def test_CandBmatrix():
    x, y = map(np.ravel, np.indices((3, 3)))
    C = fitting.Cmatrix(x, y, sx=1, sy=2, theta=0)
    assert not np.any(np.isnan(C))
    B = fitting.Bmatrix(C)
    assert not np.any(np.isnan(C))


def test_hessian_shape():
    # test a single component model
    model = lmfit.Parameters()
    model.add('c0_amp', 1, vary=True)
    model.add('c0_xo', 5, vary=True)
    model.add('c0_yo', 5, vary=True)
    model.add('c0_sx', 2.001, vary=False)
    model.add('c0_sy', 2, vary=False)
    model.add('c0_theta', 0, vary=False)
    model.add('components', 1, vary=False)
    nvar = 3
    x, y = np.indices((10, 10))
    Hij = fitting.hessian(model, x, y)
    assert Hij.shape == (nvar, nvar, 10, 10)

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
    assert Hij.shape == (nvar, nvar, 10, 10)


def test_jacobian_shape():
    """
    Test to see if the Jacobian matrix if of the right shape
    This includes a single source model with only 4 variable params
    :return: True if the test passes
    """
    model = lmfit.Parameters()
    model.add('c0_amp', 1, vary=True)
    model.add('c0_xo', 5, vary=True)
    model.add('c0_yo', 5, vary=True)
    model.add('c0_sx', 2.001, vary=False)
    model.add('c0_sy', 2, vary=False)
    model.add('c0_theta', 0, vary=False)
    model.add('components', 1, vary=False)
    nvar = 3
    x, y = np.indices((10, 10))
    Jij = fitting.jacobian(model, x, y)
    assert Jij.shape == (nvar, 10, 10)

    model.add('c1_amp', 1, vary=True)
    model.add('c1_xo', 5, vary=True)
    model.add('c1_yo', 5, vary=True)
    model.add('c1_sx', 2.001, vary=True)
    model.add('c1_sy', 2, vary=True)
    model.add('c1_theta', 0, vary=True)
    nvar = 9
    model['components'].value = 2
    Jij = fitting.jacobian(model, x, y)
    assert Jij.shape == (nvar, 10, 10)


def test_emp_vs_ana_jacobian():
    model = lmfit.Parameters()
    model.add('c0_amp', 1, vary=True)
    model.add('c0_xo', 5, vary=True)
    model.add('c0_yo', 5, vary=True)
    model.add('c0_sx', 2.001, vary=False)
    model.add('c0_sy', 2, vary=False)
    model.add('c0_theta', 0, vary=False)
    model.add('components', 1, vary=False)
    nvar = 3
    x, y = np.indices((10, 10))
    emp_Jij = fitting.emp_jacobian(model, x, y)
    ana_Jij = fitting.jacobian(model, x, y)
    diff = np.abs(ana_Jij - emp_Jij)
    assert np.max(diff) < 1e-5

    model.add('c1_amp', 1, vary=True)
    model.add('c1_xo', 5, vary=True)
    model.add('c1_yo', 5, vary=True)
    model.add('c1_sx', 2.001, vary=True)
    model.add('c1_sy', 2, vary=True)
    model.add('c1_theta', 0, vary=True)
    nvar = 9
    model['components'].value = 2
    emp_Jij = fitting.emp_jacobian(model, x, y)
    ana_Jij = fitting.jacobian(model, x, y)
    diff = np.abs(ana_Jij - emp_Jij)
    assert np.max(diff) < 1e-3


def test_emp_vs_ana_hessian():
    model = lmfit.Parameters()
    model.add('c0_amp', 1, vary=True)
    model.add('c0_xo', 5, vary=True)
    model.add('c0_yo', 5, vary=True)
    model.add('c0_sx', 2.001, vary=False)
    model.add('c0_sy', 2, vary=False)
    model.add('c0_theta', 0, vary=False)
    model.add('components', 1, vary=False)
    nvar = 3
    x, y = np.indices((10, 10))
    emp_Hij = fitting.emp_hessian(model, x, y)
    ana_Hij = fitting.hessian(model, x, y)
    diff = np.abs(ana_Hij - emp_Hij)
    assert np.max(diff) < 1e-5

    model.add('c1_amp', 1, vary=True)
    model.add('c1_xo', 5, vary=True)
    model.add('c1_yo', 5, vary=True)
    model.add('c1_sx', 2.001, vary=True)
    model.add('c1_sy', 2, vary=True)
    model.add('c1_theta', 0, vary=True)
    nvar = 9
    model['components'].value = 2
    emp_Hij = fitting.emp_hessian(model, x, y)
    ana_Hij = fitting.hessian(model, x, y)
    diff = np.abs(ana_Hij - emp_Hij)
    assert np.max(diff) < 1


def test_make_ita():
    noise = np.random.random((10, 10))
    ita = fitting.make_ita(noise)
    assert ita.shape == (100, 100)
    noise *= np.nan
    ita = fitting.make_ita(noise)
    assert len(ita) == 0


def test_RB_bias():
    data = np.random.random((4, 4))
    model = make_model()
    bias = fitting.RB_bias(data, model)
    assert len(bias) == 3


def test_bias_correct():
    data = np.random.random((4, 4))
    model = make_model()
    fitting.bias_correct(model, data)


def dont_test_CRB_errs():
    x, y = np.indices((3, 3))
    C = fitting.Cmatrix(x.ravel(), y.ravel(), sx=1, sy=2, theta=0)
    model = make_model()
    Jij = fitting.jacobian(model, x, y)
    fitting.CRB_errs(Jij, C)
    B = fitting.Bmatrix(C)
    fitting.CRB_errs(Jij, C, B)


def test_condon_errs():
    source = models.OutputSource()
    source.ra = 0
    source.dec = 1
    source.a = 10
    source.b = 10.
    source.pa = 0
    source.local_rms = 0.1
    source.peak_flux = 1
    source.int_flux = 1
    fitting.condon_errors(source, None)
    assert source.err_a == 0
    fitting.condon_errors(source, theta_n=8.)
    assert source.err_a > 0


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
