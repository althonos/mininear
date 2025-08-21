from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from types import ModuleType
    from typing import Annotated, Literal, TypeVar
    import numpy.typing

    DType = TypeVar("DType", bound=numpy.generic)
    ArrayN = Annotated[numpy.typing.NDArray[DType], Literal["N"]]
    ArrayNx256 = Annotated[numpy.typing.NDArray[DType], Literal["N", 256]]
    ArrayNxM = Annotated[numpy.typing.NDArray[DType], Literal["N", "M"]]