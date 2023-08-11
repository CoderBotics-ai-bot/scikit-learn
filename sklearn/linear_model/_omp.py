"""Orthogonal matching pursuit algorithms
"""
from math import sqrt
from numbers import Integral, Real

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs

from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..model_selection import check_cv
from ..utils import as_float_array, check_array
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, _deprecate_normalize, _pre_fit


from typing import Union, Optional, Tuple
import warnings

# Author: Vlad Niculae
#
# License: BSD 3 clause

import warnings

premature = (
    "Orthogonal matching pursuit ended prematurely due to linear"
    " dependence in the dictionary. The requested precision might"
    " not have been met."
)


def _cholesky_omp(X, y, n_nonzero_coefs, tol=None, copy_X=True, return_path=False):
    X, y, n_features, L, residuals = _validate_and_initialize(
        X, y, n_nonzero_coefs, tol, copy_X
    )
    gamma, idx, active_set, L = _omp_main_loop(
        X, y, n_nonzero_coefs, tol, residuals, L, norms, least_squares
    )
    coef = _return_results(return_path, idx, gamma, L, active_set, coef)

    if tol is not None and np.sum(residuals**2) > tol:
        warnings.warn(premature, RuntimeError, stacklevel=2)

    return gamma, idx, coef


def _gram_omp(
    Gram: np.ndarray,
    Xy: np.ndarray,
    n_nonzero_coefs: int,
    tol_0: Optional[float] = None,
    tol: Optional[float] = None,
    copy_Gram: bool = True,
    copy_Xy: bool = True,
    return_path: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray, int]
]:
    _validate_tol(tol, tol_0, n_nonzero_coefs)

    if copy_Gram:
        Gram = Gram.copy("F")
    if copy_Xy:
        Xy = Xy.copy()

    # current implementation stays as it is...

def _validate_and_initialize(X, y, n_nonzero_coefs, tol, copy_X):
    if copy_X:
        X = X.copy("F")
    if check_array:
        y = y.copy()
    n_features = X.shape[1]
    if tol is not None:
        max_features = min(max_features, 1 + np.sum(norms > tol))
    L = np.empty((n_features, n_features), dtype=X.dtype)
    residuals = y
    return X, y, n_features, L, residuals

def _validate_tol(
    tol: Optional[float], tol_0: Optional[float], n_nonzero_coefs: int
) -> None:
    """Check if `tol` is valid based on `tol_0`.
    Parameters
    ----------
    tol : Optional[float]
        The targeted squared error. If provided, it overrides n_nonzero_coefs.
    tol_0 : Optional[float]
        The squared norm of y, required if tol is not None.
    n_nonzero_coefs : int
        The targeted number of non-zero elements in the solution.
    """
    if tol is not None and tol_0 is None:
        raise ValueError("tol_0 must be set if tol is not None.")
    if (
        tol is None
        and tol_0 is None
        and np.sum(Xy**2) < (n_nonzero_coefs - 1) * np.finfo(Xy.dtype).eps
    ):
        warnings.warn(premature, RuntimeWarning)


def _omp_main_loop(X, y, n_nonzero_coefs, tol, residuals, L, norms, least_squares):
    active_set = np.zeros(n_features, dtype=bool)
    gamma = np.zeros(n_nonzero_coefs, dtype=X.dtype)
    idx = np.empty(n_nonzero_coefs, dtype=int)
    for n_active in range(n_nonzero_coefs):
        # code ...
        # Here is the main loop of Orthogonal Matching Pursuit algorithm.

        coef = np.empty((n_features, n_nonzero_coefs), dtype=X.dtype)
        coef[active_set, :n_active] = L[:n_active, :n_active].T
        coef[~active_set, :n_active] = 0
    else:
        coef = np.zeros(n_features, dtype=X.dtype)
        coef[idx[:n_active]] = gamma
    return coef


@validate_params(
    {
        "X": ["array-like"],
        "y": [np.ndarray],
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "precompute": ["boolean", StrOptions({"auto"})],
        "copy_X": ["boolean"],
        "return_path": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def orthogonal_mp(
    X,
    y,
    *,
    n_nonzero_coefs=None,
    tol=None,
    precompute=False,
    copy_X=True,
    return_path=False,
    return_n_iter=False,
):
    r"""Orthogonal Matching Pursuit (OMP).

    Solves n_targets Orthogonal Matching Pursuit problems.
    An instance of the problem has the form:

    When parametrized by the number of non-zero coefficients using
    `n_nonzero_coefs`:
    argmin ||y - X\gamma||^2 subject to ||\gamma||_0 <= n_{nonzero coefs}

    When parametrized by error using the parameter `tol`:
    argmin ||\gamma||_0 subject to ||y - X\gamma||^2 <= tol

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data. Columns are assumed to have unit norm.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Input targets.

    n_nonzero_coefs : int, default=None
        Desired number of non-zero entries in the solution. If None (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum squared norm of the residual. If not None, overrides n_nonzero_coefs.

    precompute : 'auto' or bool, default=False
        Whether to perform precomputations. Improves performance when n_targets
        or n_samples is very large.

    copy_X : bool, default=True
        Whether the design matrix X must be copied by the algorithm. A false
        value is only helpful if X is already Fortran-ordered, otherwise a
        copy is made anyway.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        Coefficients of the OMP solution. If `return_path=True`, this contains
        the whole coefficient path. In this case its shape is
        (n_features, n_features) or (n_features, n_targets, n_features) and
        iterating over the last axis generates coefficients in increasing order
        of active features.

    n_iters : array-like or int
        Number of active features across every target. Returned only if
        `return_n_iter` is set to True.

    See Also
    --------
    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model.
    orthogonal_mp_gram : Solve OMP problems using Gram matrix and the product X.T * y.
    lars_path : Compute Least Angle Regression or Lasso path using LARS algorithm.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Notes
    -----
    Orthogonal matching pursuit was introduced in S. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    (https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf)

    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
    """
    X = check_array(X, order="F", copy=copy_X)
    copy_X = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y = check_array(y)
    if y.shape[1] > 1:  # subsequent targets will be affected
        copy_X = True
    if n_nonzero_coefs is None and tol is None:
        # default for n_nonzero_coefs is 0.1 * n_features
        # but at least one.
        n_nonzero_coefs = max(int(0.1 * X.shape[1]), 1)
    if tol is None and n_nonzero_coefs > X.shape[1]:
        raise ValueError(
            "The number of atoms cannot be more than the number of features"
        )
    if precompute == "auto":
        precompute = X.shape[0] > X.shape[1]
    if precompute:
        G = np.dot(X.T, X)
        G = np.asfortranarray(G)
        Xy = np.dot(X.T, y)
        if tol is not None:
            norms_squared = np.sum((y**2), axis=0)
        else:
            norms_squared = None
        return orthogonal_mp_gram(
            G,
            Xy,
            n_nonzero_coefs=n_nonzero_coefs,
            tol=tol,
            norms_squared=norms_squared,
            copy_Gram=copy_X,
            copy_Xy=False,
            return_path=return_path,
        )

    if return_path:
        coef = np.zeros((X.shape[1], y.shape[1], X.shape[1]))
    else:
        coef = np.zeros((X.shape[1], y.shape[1]))
    n_iters = []

    for k in range(y.shape[1]):
        out = _cholesky_omp(
            X, y[:, k], n_nonzero_coefs, tol, copy_X=copy_X, return_path=return_path
        )
        if return_path:
            _, idx, coefs, n_iter = out
            coef = coef[:, :, : len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[: n_active + 1], k, n_active] = x[: n_active + 1]
        else:
            x, idx, n_iter = out
            coef[idx, k] = x
        n_iters.append(n_iter)

    if y.shape[1] == 1:
        n_iters = n_iters[0]

    if return_n_iter:
        return np.squeeze(coef), n_iters
    else:
        return np.squeeze(coef)


@validate_params(
    {
        "Gram": ["array-like"],
        "Xy": ["array-like"],
        "n_nonzero_coefs": [Interval(Integral, 0, None, closed="neither"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "norms_squared": ["array-like", None],
        "copy_Gram": ["boolean"],
        "copy_Xy": ["boolean"],
        "return_path": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def orthogonal_mp_gram(
    Gram,
    Xy,
    *,
    n_nonzero_coefs=None,
    tol=None,
    norms_squared=None,
    copy_Gram=True,
    copy_Xy=True,
    return_path=False,
    return_n_iter=False,
):
    """Gram Orthogonal Matching Pursuit (OMP).

    Solves n_targets Orthogonal Matching Pursuit problems using only
    the Gram matrix X.T * X and the product X.T * y.

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    Gram : array-like of shape (n_features, n_features)
        Gram matrix of the input data: `X.T * X`.

    Xy : array-like of shape (n_features,) or (n_features, n_targets)
        Input targets multiplied by `X`: `X.T * y`.

    n_nonzero_coefs : int, default=None
        Desired number of non-zero entries in the solution. If `None` (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum squared norm of the residual. If not `None`,
        overrides `n_nonzero_coefs`.

    norms_squared : array-like of shape (n_targets,), default=None
        Squared L2 norms of the lines of `y`. Required if `tol` is not None.

    copy_Gram : bool, default=True
        Whether the gram matrix must be copied by the algorithm. A `False`
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, default=True
        Whether the covariance vector `Xy` must be copied by the algorithm.
        If `False`, it may be overwritten.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        Coefficients of the OMP solution. If `return_path=True`, this contains
        the whole coefficient path. In this case its shape is
        `(n_features, n_features)` or `(n_features, n_targets, n_features)` and
        iterating over the last axis yields coefficients in increasing order
        of active features.

    n_iters : list or int
        Number of active features across every target. Returned only if
        `return_n_iter` is set to True.

    See Also
    --------
    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model (OMP).
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    lars_path : Compute Least Angle Regression or Lasso path using
        LARS algorithm.
    sklearn.decomposition.sparse_encode : Generic sparse coding.
        Each column of the result is the solution to a Lasso problem.

    Notes
    -----
    Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    (https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf)

    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
    """
    Gram = check_array(Gram, order="F", copy=copy_Gram)
    Xy = np.asarray(Xy)
    if Xy.ndim > 1 and Xy.shape[1] > 1:
        # or subsequent target will be affected
        copy_Gram = True
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]
        if tol is not None:
            norms_squared = [norms_squared]
    if copy_Xy or not Xy.flags.writeable:
        # Make the copy once instead of many times in _gram_omp itself.
        Xy = Xy.copy()

    if n_nonzero_coefs is None and tol is None:
        n_nonzero_coefs = int(0.1 * len(Gram))
    if tol is not None and norms_squared is None:
        raise ValueError(
            "Gram OMP needs the precomputed norms in order "
            "to evaluate the error sum of squares."
        )
    if tol is not None and tol < 0:
        raise ValueError("Epsilon cannot be negative")
    if tol is None and n_nonzero_coefs <= 0:
        raise ValueError("The number of atoms must be positive")
    if tol is None and n_nonzero_coefs > len(Gram):
        raise ValueError(
            "The number of atoms cannot be more than the number of features"
        )

    if return_path:
        coef = np.zeros((len(Gram), Xy.shape[1], len(Gram)), dtype=Gram.dtype)
    else:
        coef = np.zeros((len(Gram), Xy.shape[1]), dtype=Gram.dtype)

    n_iters = []
    for k in range(Xy.shape[1]):
        out = _gram_omp(
            Gram,
            Xy[:, k],
            n_nonzero_coefs,
            norms_squared[k] if tol is not None else None,
            tol,
            copy_Gram=copy_Gram,
            copy_Xy=False,
            return_path=return_path,
        )
        if return_path:
            _, idx, coefs, n_iter = out
            coef = coef[:, :, : len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[: n_active + 1], k, n_active] = x[: n_active + 1]
        else:
            x, idx, n_iter = out
            coef[idx, k] = x
        n_iters.append(n_iter)

    if Xy.shape[1] == 1:
        n_iters = n_iters[0]

    if return_n_iter:
        return np.squeeze(coef), n_iters
    else:
        return np.squeeze(coef)


class OrthogonalMatchingPursuit(MultiOutputMixin, RegressorMixin, LinearModel):
    """Orthogonal Matching Pursuit model (OMP).

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    n_nonzero_coefs : int, default=None
        Desired number of non-zero entries in the solution. If None (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum squared norm of the residual. If not None, overrides n_nonzero_coefs.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. versionchanged:: 1.2
           default changed from True to False in 1.2.

        .. deprecated:: 1.2
            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.

    precompute : 'auto' or bool, default='auto'
        Whether to use a precomputed Gram and Xy matrix to speed up
        calculations. Improves performance when :term:`n_targets` or
        :term:`n_samples` is very large. Note that if you already have such
        matrices, you can pass them directly to the fit method.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formula).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : int or array-like
        Number of active features across every target.

    n_nonzero_coefs_ : int
        The number of non-zero coefficients in the solution. If
        `n_nonzero_coefs` is None and `tol` is None this value is either set
        to 10% of `n_features` or 1, whichever is greater.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    orthogonal_mp_gram :  Solves n_targets Orthogonal Matching Pursuit
        problems using only the Gram matrix X.T * X and the product X.T * y.
    lars_path : Compute Least Angle Regression or Lasso path using LARS algorithm.
    Lars : Least Angle Regression model a.k.a. LAR.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    sklearn.decomposition.sparse_encode : Generic sparse coding.
        Each column of the result is the solution to a Lasso problem.
    OrthogonalMatchingPursuitCV : Cross-validated
        Orthogonal Matching Pursuit model (OMP).

    Notes
    -----
    Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    (https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf)

    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    Examples
    --------
    >>> from sklearn.linear_model import OrthogonalMatchingPursuit
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> reg = OrthogonalMatchingPursuit().fit(X, y)
    >>> reg.score(X, y)
    0.9991...
    >>> reg.predict(X[:1,])
    array([-78.3854...])
    """

    _parameter_constraints: dict = {
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "fit_intercept": ["boolean"],
        "normalize": ["boolean", Hidden(StrOptions({"deprecated"}))],
        "precompute": [StrOptions({"auto"}), "boolean"],
    }

    def __init__(
        self,
        *,
        n_nonzero_coefs=None,
        tol=None,
        fit_intercept=True,
        normalize="deprecated",
        precompute="auto",
    ):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        _normalize = _deprecate_normalize(
            self.normalize, estimator_name=self.__class__.__name__
        )

        X, y = self._validate_data(X, y, multi_output=True, y_numeric=True)
        n_features = X.shape[1]

        X, y, X_offset, y_offset, X_scale, Gram, Xy = _pre_fit(
            X, y, None, self.precompute, _normalize, self.fit_intercept, copy=True
        )

        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.n_nonzero_coefs is None and self.tol is None:
            # default for n_nonzero_coefs is 0.1 * n_features
            # but at least one.
            self.n_nonzero_coefs_ = max(int(0.1 * n_features), 1)
        else:
            self.n_nonzero_coefs_ = self.n_nonzero_coefs

        if Gram is False:
            coef_, self.n_iter_ = orthogonal_mp(
                X,
                y,
                n_nonzero_coefs=self.n_nonzero_coefs_,
                tol=self.tol,
                precompute=False,
                copy_X=True,
                return_n_iter=True,
            )
        else:
            norms_sq = np.sum(y**2, axis=0) if self.tol is not None else None

            coef_, self.n_iter_ = orthogonal_mp_gram(
                Gram,
                Xy=Xy,
                n_nonzero_coefs=self.n_nonzero_coefs_,
                tol=self.tol,
                norms_squared=norms_sq,
                copy_Gram=True,
                copy_Xy=True,
                return_n_iter=True,
            )
        self.coef_ = coef_.T
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


def _omp_path_residues(
    X_train,
    y_train,
    X_test,
    y_test,
    copy=True,
    fit_intercept=True,
    normalize=False,
    max_iter=100,
):
    """Compute the residues on left-out data for a full LARS path.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        The data to fit the LARS on.

    y_train : ndarray of shape (n_samples)
        The target variable to fit LARS on.

    X_test : ndarray of shape (n_samples, n_features)
        The data to compute the residues on.

    y_test : ndarray of shape (n_samples)
        The target variable to compute the residues on.

    copy : bool, default=True
        Whether X_train, X_test, y_train and y_test should be copied.  If
        False, they may be overwritten.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. versionchanged:: 1.2
           default changed from True to False in 1.2.

        .. deprecated:: 1.2
            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.

    max_iter : int, default=100
        Maximum numbers of iterations to perform, therefore maximum features
        to include. 100 by default.

    Returns
    -------
    residues : ndarray of shape (n_samples, max_features)
        Residues of the prediction on the test data.
    """

    if copy:
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_test = X_test.copy()
        y_test = y_test.copy()

    if fit_intercept:
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean
        y_mean = y_train.mean(axis=0)
        y_train = as_float_array(y_train, copy=False)
        y_train -= y_mean
        y_test = as_float_array(y_test, copy=False)
        y_test -= y_mean

    if normalize:
        norms = np.sqrt(np.sum(X_train**2, axis=0))
        nonzeros = np.flatnonzero(norms)
        X_train[:, nonzeros] /= norms[nonzeros]

    coefs = orthogonal_mp(
        X_train,
        y_train,
        n_nonzero_coefs=max_iter,
        tol=None,
        precompute=False,
        copy_X=False,
        return_path=True,
    )
    if coefs.ndim == 1:
        coefs = coefs[:, np.newaxis]
    if normalize:
        coefs[nonzeros] /= norms[nonzeros][:, np.newaxis]

    return np.dot(coefs.T, X_test.T) - y_test


class OrthogonalMatchingPursuitCV(RegressorMixin, LinearModel):
    """Cross-validated Orthogonal Matching Pursuit model (OMP).

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    copy : bool, default=True
        Whether the design matrix X must be copied by the algorithm. A false
        value is only helpful if X is already Fortran-ordered, otherwise a
        copy is made anyway.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. versionchanged:: 1.2
           default changed from True to False in 1.2.

        .. deprecated:: 1.2
            ``normalize`` was deprecated in version 1.2 and will be removed in 1.4.

    max_iter : int, default=None
        Maximum numbers of iterations to perform, therefore maximum features
        to include. 10% of ``n_features`` but at least 5 if available.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool or int, default=False
        Sets the verbosity amount.

    Attributes
    ----------
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the problem formulation).

    n_nonzero_coefs_ : int
        Estimated number of non-zero coefficients giving the best mean squared
        error over the cross-validation folds.

    n_iter_ : int or array-like
        Number of active features across every target for the model refit with
        the best hyperparameters got by cross-validating across all folds.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    orthogonal_mp_gram : Solves n_targets Orthogonal Matching Pursuit
        problems using only the Gram matrix X.T * X and the product X.T * y.
    lars_path : Compute Least Angle Regression or Lasso path using LARS algorithm.
    Lars : Least Angle Regression model a.k.a. LAR.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model (OMP).
    LarsCV : Cross-validated Least Angle Regression model.
    LassoLarsCV : Cross-validated Lasso model fit with Least Angle Regression.
    sklearn.decomposition.sparse_encode : Generic sparse coding.
        Each column of the result is the solution to a Lasso problem.

    Notes
    -----
    In `fit`, once the optimal number of non-zero coefficients is found through
    cross-validation, the model is fit again using the entire training set.

    Examples
    --------
    >>> from sklearn.linear_model import OrthogonalMatchingPursuitCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=100, n_informative=10,
    ...                        noise=4, random_state=0)
    >>> reg = OrthogonalMatchingPursuitCV(cv=5).fit(X, y)
    >>> reg.score(X, y)
    0.9991...
    >>> reg.n_nonzero_coefs_
    10
    >>> reg.predict(X[:1,])
    array([-78.3854...])
    """

    _parameter_constraints: dict = {
        "copy": ["boolean"],
        "fit_intercept": ["boolean"],
        "normalize": ["boolean", Hidden(StrOptions({"deprecated"}))],
        "max_iter": [Interval(Integral, 0, None, closed="left"), None],
        "cv": ["cv_object"],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        copy=True,
        fit_intercept=True,
        normalize="deprecated",
        max_iter=None,
        cv=None,
        n_jobs=None,
        verbose=False,
    ):
        self.copy = copy
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        _normalize = _deprecate_normalize(
            self.normalize, estimator_name=self.__class__.__name__
        )

        X, y = self._validate_data(X, y, y_numeric=True, ensure_min_features=2)
        X = as_float_array(X, copy=False, force_all_finite=False)
        cv = check_cv(self.cv, classifier=False)
        max_iter = (
            min(max(int(0.1 * X.shape[1]), 5), X.shape[1])
            if not self.max_iter
            else self.max_iter
        )
        cv_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_omp_path_residues)(
                X[train],
                y[train],
                X[test],
                y[test],
                self.copy,
                self.fit_intercept,
                _normalize,
                max_iter,
            )
            for train, test in cv.split(X)
        )

        min_early_stop = min(fold.shape[0] for fold in cv_paths)
        mse_folds = np.array(
            [(fold[:min_early_stop] ** 2).mean(axis=1) for fold in cv_paths]
        )
        best_n_nonzero_coefs = np.argmin(mse_folds.mean(axis=0)) + 1
        self.n_nonzero_coefs_ = best_n_nonzero_coefs
        omp = OrthogonalMatchingPursuit(
            n_nonzero_coefs=best_n_nonzero_coefs,
            fit_intercept=self.fit_intercept,
            normalize=_normalize,
        )

        # avoid duplicating warning for deprecated normalize
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            omp.fit(X, y)

        self.coef_ = omp.coef_
        self.intercept_ = omp.intercept_
        self.n_iter_ = omp.n_iter_
        return self
