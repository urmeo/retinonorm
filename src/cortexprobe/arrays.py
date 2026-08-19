"""Array type aliases.

Every module here passes NumPy arrays around, and each one used to declare its own
``FloatArray = np.ndarray``. That alias says nothing: ``np.ndarray`` is generic in its shape
and dtype, so mypy resolves every element access to ``Any`` and quietly stops checking the
numerics -- which is the part of this codebase most worth checking. Naming the dtype once,
here, restores that.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int_]
