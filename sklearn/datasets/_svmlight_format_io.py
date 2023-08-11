"""This module implements a loader and dumper for the svmlight format

This format is a text-based format, with one sample per line. It does
not store zero valued features hence is suitable for sparse dataset.

The first element of each line can be used to store a target variable to
predict.

This format is used as the default format for both svmlight and the
libsvm command line programs.
"""
from contextlib import closing
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from .. import __version__
from ..utils import IS_PYPY, check_array
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params


import os
from typing import Any, Union


import gzip
import bz2
from typing import Union, Any

# Authors: Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck
#          Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

import os.path

if not IS_PYPY:
    from ._svmlight_format_fast import (
        _dump_svmlight_file,
        _load_svmlight_file,
    )
else:

    def _load_svmlight_file(*args, **kwargs):
        raise NotImplementedError(
            "load_svmlight_file is currently not "
            "compatible with PyPy (see "
            "https://github.com/scikit-learn/scikit-learn/issues/11543 "
            "for the status updates)."
        )


def load_svmlight_files(
    files,
    *,
    n_features=None,
    dtype=np.float64,
    multilabel=False,
    zero_based="auto",
    query_id=False,
    offset=0,
    length=-1,
) -> list:
    """Load one or more files in SVMlight format.

    Parameters and Return are same as before.
    """
    r = []
    zero_based = bool(zero_based)

    for i, f in enumerate(files):
        result = process_single_file(
            f,
            dtype=dtype,
            multilabel=multilabel,
            zero_based=zero_based,
            query_id=query_id,
            offset=offset,
            length=length,
        )

        if n_features is None and i == 0:
            n_features = result[0].shape[1]

        elif result[0].shape[1] != n_features:
            raise ValueError(
                "Inconsistent n_features: %d != %d" % (result[0].shape[1], n_features)
            )

        r.extend(result)

    return r


@validate_params(
    {
        "f": [
            str,
            Interval(Integral, 0, None, closed="left"),
            os.PathLike,
            HasMethods("read"),
        ],
        "n_features": [Interval(Integral, 1, None, closed="left"), None],
        "dtype": "no_validation",  # delegate validation to numpy
        "multilabel": ["boolean"],
        "zero_based": ["boolean", StrOptions({"auto"})],
        "query_id": ["boolean"],
        "offset": [Interval(Integral, 0, None, closed="left")],
        "length": [Integral],
    },
    prefer_skip_nested_validation=True,
)
def load_svmlight_file(
    f,
    *,
    n_features=None,
    dtype=np.float64,
    multilabel=False,
    zero_based="auto",
    query_id=False,
    offset=0,
    length=-1,
):
    """Load datasets in the svmlight / libsvm format into sparse CSR matrix.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    This format is used as the default format for both svmlight and the
    libsvm command line programs.

    Parsing a text based source can be expensive. When repeatedly
    working on the same dataset, it is recommended to wrap this
    loader with joblib.Memory.cache to store a memmapped backup of the
    CSR results of the first call and benefit from the near instantaneous
    loading of memmapped structures for the subsequent calls.

    In case the file contains a pairwise preference constraint (known
    as "qid" in the svmlight format) these are ignored unless the
    query_id parameter is set to True. These pairwise preference
    constraints can be used to constraint the combination of samples
    when using pairwise loss functions (as is the case in some
    learning to rank problems) so that only pairs with the same
    query_id value are considered.

    This implementation is written in Cython and is reasonably fast.
    However, a faster API-compatible loader is also available at:

      https://github.com/mblondel/svmlight-loader

    Parameters
    ----------
    f : str, path-like, file-like or int
        (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. A file-like or file descriptor will not be closed
        by this function. A file-like object must be opened in binary mode.

        .. versionchanged:: 1.2
           Path-like objects are now accepted.

    n_features : int, default=None
        The number of features to use. If None, it will be inferred. This
        argument is useful to load several files that are subsets of a
        bigger sliced dataset: each subset might not have examples of
        every feature, hence the inferred shape might vary from one
        slice to another.
        n_features is only required if ``offset`` or ``length`` are passed a
        non-default value.

    dtype : numpy data type, default=np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).

    zero_based : bool or "auto", default="auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no ``offset`` or ``length`` is passed.
        If ``offset`` or ``length`` are passed, the "auto" mode falls back
        to ``zero_based=True`` to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : bool, default=False
        If True, will return the query_id array for each file.

    offset : int, default=0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.

    length : int, default=-1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.

    Returns
    -------
    X : scipy.sparse matrix of shape (n_samples, n_features)
        The data matrix.

    y : ndarray of shape (n_samples,), or a list of tuples of length n_samples
        The target. It is a list of tuples when ``multilabel=True``, else a
        ndarray.

    query_id : array of shape (n_samples,)
       The query_id for each sample. Only returned when query_id is set to
       True.

    See Also
    --------
    load_svmlight_files : Similar function for loading multiple files in this
        format, enforcing the same number of features/columns on all of them.

    Examples
    --------
    To use joblib.Memory to cache the svmlight file::

        from joblib import Memory
        from .datasets import load_svmlight_file
        mem = Memory("./mycache")

        @mem.cache
        def get_data():
            data = load_svmlight_file("mysvmlightfile")
            return data[0], data[1]

        X, y = get_data()
    """
    return tuple(
        load_svmlight_files(
            [f],
            n_features=n_features,
            dtype=dtype,
            multilabel=multilabel,
            zero_based=zero_based,
            query_id=query_id,
            offset=offset,
            length=length,
        )
    )

def validate_file(file: Union[str, int, Any]) -> Any:
    """Validate the given file.

    This helper function is used to determine the right file opening function for the file.
    If the filename ends with '.gz' or '.bz2', the file is decompressed.

    Parameters
    ----------
    file : str, file descriptor or file-like
        The file path or file-like object.

    Returns
    -------
    file : file-like object
        The opened file object. Can also be a compressed file.
    """
    file_ext = os.path.splitext(file)[1]
    if file_ext == ".gz":
        open_file = gzip.open
    elif file_ext == ".bz2":
        open_file = bz2.open
    else:
        open_file = open

    return open_file(file, "rb")


def process_single_file(file: Union[str, int, Any], **kwargs) -> Any:
    """Process a single file data.

    This function opens the file if it is not already opened and then
    load its contents.

    Parameters
    ----------
    file : str, file descriptor or file-like
        The file path or file-like object.

    **kwargs : dict
        Keyword arguments are passed to '_load_svmlight_file' function.

    Returns
    -------
    tuple
        The loaded file data.
    """
    if isinstance(file, str):
        with closing(validate_file(file)) as f:
            return _load_svmlight_file(f, **kwargs)

    return _load_svmlight_file(file, **kwargs)

def _gen_open(f: Union[int, os.PathLike, str]) -> Any:
    """
    Open various types of files (including compressed files) for reading.

    This function is primarily used for reading files in the SVMLight format. The types of files it
    can handle include file descriptors (specified by an integer), PathLike objects and uncompressed or
    gzip/bz2 compressed files (both specified by a string).

    Parameters
    ----------
    f : Union[int, os.PathLike, str]
        File to open. Could either be a file descriptor, a PathLike object, or a directly specified uncompressed
        or gzip/bz2 compressed file.

    Returns
    -------
    Any
        A file object corresponding to the opened file.

    Raises
    ------
    TypeError
        If the input is not an integer, a PathLike object, or a string pointing to a file.

    """

    if isinstance(f, str):
        f = os.path.expanduser(f)
        return _open_compressed_file(f)

    elif isinstance(f, Integral):
        return os.fdopen(f, "rb")

    elif isinstance(f, os.PathLike):
        return open(f, "rb")

    else:
        raise TypeError("Input should be a string, integer or os.PathLike object.")


def _open_and_load(f, dtype, multilabel, zero_based, query_id, offset=0, length=-1):
    if hasattr(f, "read"):
        actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(
            f, dtype, multilabel, zero_based, query_id, offset, length
        )
    else:
        with closing(_gen_open(f)) as f:
            actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(
                f, dtype, multilabel, zero_based, query_id, offset, length
            )

    # convert from array.array, give data the right dtype
    if not multilabel:
        labels = np.frombuffer(labels, np.float64)
    data = np.frombuffer(data, actual_dtype)
    indices = np.frombuffer(ind, np.longlong)
    indptr = np.frombuffer(indptr, dtype=np.longlong)  # never empty
    query = np.frombuffer(query, np.int64)

    data = np.asarray(data, dtype=dtype)  # no-op for float{32,64}
    return data, indices, indptr, labels, query


def _open_compressed_file(filename: str) -> Any:
    """
    Opens a compressed file for reading.

    Parameters
    ----------
    filename : str
        Name of the file to open.

    Returns
    -------
    Any
        File object corresponding to the opened file.

    """

    if filename.endswith(".gz"):
        return gzip.open(filename, "rb")
    elif filename.endswith(".bz2"):
        return bz2.open(filename, "rb")
    else:
        return open(filename, "rb")


def _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id):
    if comment:
        f.write(
            (
                "# Generated by dump_svmlight_file from scikit-learn %s\n" % __version__
            ).encode()
        )
        f.write(
            ("# Column indices are %s-based\n" % ["zero", "one"][one_based]).encode()
        )

        f.write(b"#\n")
        f.writelines(b"# %s\n" % line for line in comment.splitlines())
    X_is_sp = sp.issparse(X)
    y_is_sp = sp.issparse(y)
    if not multilabel and not y_is_sp:
        y = y[:, np.newaxis]
    _dump_svmlight_file(
        X,
        y,
        f,
        multilabel,
        one_based,
        query_id,
        X_is_sp,
        y_is_sp,
    )


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "y": ["array-like", "sparse matrix"],
        "f": [str, HasMethods(["write"])],
        "zero_based": ["boolean"],
        "comment": [str, bytes, None],
        "query_id": ["array-like", None],
        "multilabel": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def dump_svmlight_file(
    X,
    y,
    f,
    *,
    zero_based=True,
    comment=None,
    query_id=None,
    multilabel=False,
):
    """Dump the dataset in svmlight / libsvm file format.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : {array-like, sparse matrix}, shape = (n_samples,) or (n_samples, n_labels)
        Target values. Class labels must be an
        integer or float, or array-like objects of integer or float for
        multilabel classifications.

    f : str or file-like in binary mode
        If string, specifies the path that will contain the data.
        If file-like, data will be written to f. f should be opened in binary
        mode.

    zero_based : bool, default=True
        Whether column indices should be written zero-based (True) or one-based
        (False).

    comment : str or bytes, default=None
        Comment to insert at the top of the file. This should be either a
        Unicode string, which will be encoded as UTF-8, or an ASCII byte
        string.
        If a comment is given, then it will be preceded by one that identifies
        the file as having been dumped by scikit-learn. Note that not all
        tools grok comments in SVMlight files.

    query_id : array-like of shape (n_samples,), default=None
        Array containing pairwise preference constraints (qid in svmlight
        format).

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).

        .. versionadded:: 0.17
           parameter `multilabel` to support multilabel datasets.
    """
    if comment is not None:
        # Convert comment string to list of lines in UTF-8.
        # If a byte string is passed, then check whether it's ASCII;
        # if a user wants to get fancy, they'll have to decode themselves.
        if isinstance(comment, bytes):
            comment.decode("ascii")  # just for the exception
        else:
            comment = comment.encode("utf-8")
        if b"\0" in comment:
            raise ValueError("comment string contains NUL byte")

    yval = check_array(y, accept_sparse="csr", ensure_2d=False)
    if sp.issparse(yval):
        if yval.shape[1] != 1 and not multilabel:
            raise ValueError(
                "expected y of shape (n_samples, 1), got %r" % (yval.shape,)
            )
    else:
        if yval.ndim != 1 and not multilabel:
            raise ValueError("expected y of shape (n_samples,), got %r" % (yval.shape,))

    Xval = check_array(X, accept_sparse="csr")
    if Xval.shape[0] != yval.shape[0]:
        raise ValueError(
            "X.shape[0] and y.shape[0] should be the same, got %r and %r instead."
            % (Xval.shape[0], yval.shape[0])
        )

    # We had some issues with CSR matrices with unsorted indices (e.g. #1501),
    # so sort them here, but first make sure we don't modify the user's X.
    # TODO We can do this cheaper; sorted_indices copies the whole matrix.
    if yval is y and hasattr(yval, "sorted_indices"):
        y = yval.sorted_indices()
    else:
        y = yval
        if hasattr(y, "sort_indices"):
            y.sort_indices()

    if Xval is X and hasattr(Xval, "sorted_indices"):
        X = Xval.sorted_indices()
    else:
        X = Xval
        if hasattr(X, "sort_indices"):
            X.sort_indices()

    if query_id is None:
        # NOTE: query_id is passed to Cython functions using a fused type on query_id.
        # Yet as of Cython>=3.0, memory views can't be None otherwise the runtime
        # would not known which concrete implementation to dispatch the Python call to.
        # TODO: simplify interfaces and implementations in _svmlight_format_fast.pyx.
        query_id = np.array([], dtype=np.int32)
    else:
        query_id = np.asarray(query_id)
        if query_id.shape[0] != y.shape[0]:
            raise ValueError(
                "expected query_id of shape (n_samples,), got %r" % (query_id.shape,)
            )

    one_based = not zero_based

    if hasattr(f, "write"):
        _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
    else:
        with open(f, "wb") as f:
            _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
