from functools import cached_property
from imp import cache_from_source
from typing import Iterable, List, Literal, Optional, SupportsInt, TypeVar, Union
import numpy as np


# from dataclasses import dataclass, field

from pydantic import Field


from .internal_data_models import BaseModel, AnnotationType


class EntityDatum(BaseModel):
    _annotationType_: AnnotationType

    node_id: Union[str, Literal["RESULT"]]
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.9664)

    @cached_property
    def sigma(self) -> float:
        return 1.0 - self.confidence

    @cached_property
    def value(self) -> np.ndarray:
        ...


class EntityShapeDatum(EntityDatum):
    @cached_property
    def xy(self) -> np.ndarray:
        ...

    @cached_property
    def area(self) -> float:
        ...


class ObjectCount(EntityDatum):
    _annotationType_ = AnnotationType.multiClassification

    count: int

    @cached_property
    def value(self):
        return self.count


class BoundingBox(EntityShapeDatum):
    _annotationType_ = AnnotationType.boundingBox

    x1: float = Field(..., ge=0.0, le=1.0, example=0.235)
    x2: float = Field(..., ge=0.0, le=1.0, example=0.340)
    y1: float = Field(..., ge=0.0, le=1.0, example=0.450)
    y2: float = Field(..., ge=0.0, le=1.0, example=0.543)

    @cached_property
    def value(self) -> np.ndarray:
        return np.array(self.tuple, dtype=np.float64)

    @cached_property
    def tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @cached_property
    def xy(self) -> np.ndarray:
        return self.value

    @cached_property
    def array(self) -> np.ndarray:
        return self.value

    @cached_property
    def coordinate_matrix(self) -> np.ndarray:
        return self.xy.reshape(2, 2)

    @cached_property
    def wh(self) -> np.ndarray:
        return np.concatenate((self.anchor_point, self.diagonal_vector))

    @cached_property
    def width(self):
        return self.x2 - self.x1

    @cached_property
    def height(self):
        return self.y2 - self.y1

    @cached_property
    def diagonal_vector(self):
        return np.array((self.width, self.height))

    @cached_property
    def diagonal_length(self):
        return np.linalg.norm(self.diagonal_vector)

    @cached_property
    def anchor_point(self):
        return np.array((self.x1, self.y1))

    @cached_property
    def area(self) -> float:
        return np.product(self.diagonal_vector)

    @cached_property
    def center(self) -> np.ndarray:
        return np.average(self.coordinate_matrix, axis=0)
