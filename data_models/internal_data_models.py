from functools import cached_property, lru_cache, partial
from typing import Any, List, Optional, Union


# from dataclasses import dataclass, field

from pydantic import Field, conlist, root_validator, validator, conint, NonNegativeInt
from pydantic import BaseModel as PydanticBaseModel
from enum import Enum
from datetime import datetime
import numpy as np

from .request_data_models import (
    RawAnnotation,
    RawBoundingBoxDatum,
    RawObjectCountDatum,
    ValidationRequestPayload,
    AnnotationType,
)


class CVSystemInternalError(RuntimeError):
    def __init__(self, message: str, payload: ValidationRequestPayload) -> None:
        self.message = f"""{'{'}"message": "Internal Error: {message}","raw_payload":"{payload.json()}"{'}'}"""
        super().__init__(self.message)


class BaseDataSet:
    ANNOTATION_TYPE: Optional[AnnotationType] = None
    POSSIBLE_ANNOTATION_TYPES = (RawObjectCountDatum, RawBoundingBoxDatum)

    def __init__(self, raw_payload: ValidationRequestPayload):
        self.raw_payload = raw_payload
        if not self.raw_payload.annotations:
            raise CVSystemInternalError(
                message="There is no annotation to validate.", payload=self.raw_payload
            )

        self._raw_node_data_list_: List[Any] = []
        self._raw_data_list_: List[Any] = []
        self._node_ids_list_: List[str] = []
        self._scores_list_: List[float] = []
        self._object_count_list_: List[int] = []

        for annotation in raw_payload.annotations:
            if annotation.annotationType != self.ANNOTATION_TYPE:
                raise CVSystemInternalError(
                    message=f"Trying to validate {annotation.annotationType} case, but it should be {self.ANNOTATION_TYPE}.",
                    payload=self.raw_payload,
                )

            assert isinstance(annotation.annotation, self.POSSIBLE_ANNOTATION_TYPES)

            self._raw_node_data_list_.append(annotation.annotation.__root__)
            self._node_ids_list_.append(annotation.userID)
            self._scores_list_.append(annotation.projectBasedConfidence)
            self._object_count_list_.append(annotation.objectCount)

    def vectorize_raw_data(self) -> Union[List[int], List[List]]:
        ...


class MultiClassificationDataSet:
    def __init__(self, raw_payload: ValidationRequestPayload) -> None:
        self.raw_payload = raw_payload
        # TODO: optimize the loop with itertools
        self._list_: List[int] = []
        self._node_ids_list_: List[str] = []
        self.node_data: List[int] = []
        assert raw_payload.annotations
        for annotation in raw_payload.annotations:
            if annotation.annotationType != AnnotationType.multiClassification:
                raise CVSystemInternalError(
                    message=f"Trying to validate {annotation.annotationType} case, but it should be {AnnotationType.multiClassification}.",
                    payload=raw_payload,
                )

            assert isinstance(
                annotation.annotation, RawObjectCountDatum
            )  # for type checker
            self.node_data.append(int(annotation.annotation.__root__))
            self._list_.append(int(annotation.annotation.__root__))
            self._node_ids_list_.append(annotation.userID)

        self.array: np.ndarray = np.array(self._list_)
        self.node_ids: np.ndarray = np.array(self._node_ids_list_)


class NodeData:
    pass


class BoundingBoxDataSet:
    def __init__(self, raw_payload: ValidationRequestPayload) -> None:
        self.raw_payload = raw_payload
        # TODO: optimize the loop
        self._list_: List[int] = []
        self._node_ids_list_: List[str] = []
        assert raw_payload.annotations
        for annotation in raw_payload.annotations:
            # TODO: logic duplication in the following if statement; factor those out to a mutual superclass
            if annotation.annotationType != AnnotationType.boundingBox:
                raise CVSystemInternalError(
                    message=f"Trying to validate {annotation.annotationType} case, but it should be {AnnotationType.multiClassification}.",
                    payload=raw_payload,
                )

            assert isinstance(
                annotation.annotation, RawBoundingBoxDatum
            )  # for type checker
            self._list_.append(annotation.annotation.__root__)
            self._node_ids_list_.append(annotation.userID)

        self.array: np.ndarray = np.array(self._list_)
        self.node_ids: np.ndarray = np.array(self._node_ids_list_)


class BaseModel(PydanticBaseModel):
    class Config:
        allow_mutation = False


class _NodeData(BaseModel):
    node_id: str
    score: float
    data: np.ndarray
    labels: np.ndarray


class ObjectData:
    def __init__(
        self, nodes: list[_NodeData], object_count: int, object_name: str
    ) -> None:
        pass


class Node:
    def __init__(self, user_name: str, score: float) -> None:
        self.name = user_name
        assert 0.0 <= score <= 1.0
        self.score = score
        self.sigma = 1 - score
        self.points: list["Data_Point"] = []

    def register(self, point: "Data_Point") -> None:
        self.points.append(point)


class Obj_Type:
    "Represents an object's class, ie. 'car', 'human', ..."

    def __init__(self, obj_name: str, _count: int) -> None:
        self.name = obj_name
        self.count = _count


class BoundingBox:
    "Main responsability of this class is to validate the coordinates of a bbox"

    def __init__(self, coordinates, kind="xy"):
        BoundingBox._validate_coordinates(coords=coordinates, kind=kind)
        self._arr = np.array(coordinates)
        assert self._arr.shape == (4,)
        self.kind = kind

    @classmethod
    def _validate_coordinates(cls, coords, kind):
        assert (
            len(coords) == 4
        ), f"Bounding box coordinates are expected to be 4 numbers. {coords=}"
        assert all(
            map(lambda _x: 0 <= _x <= 1.0, coords)
        ), f"All members of the coordinates vector should be floats between 0 and 1. {coords=}"
        assert kind in (
            "xy",
            "wh",
        ), f"Internal Error: bbox has two kinds of representation: 'xy' and 'wh', but it is given as '{kind}'"
        if kind == "xy":
            assert (
                coords[1] >= coords[0] and coords[3] >= coords[2]
            ), f"x2<x1 or y2<y1. Expected coordinates to be of increasing order. x1={coords[0]}, x2={coords[1]}, y1={coords[2]}, y2={coords[3]}"

    @cached_property
    def xy(self):
        if self.kind == "xy":
            return self._arr.copy()
        # self._arr is of the form x,y,w,h
        _xy = self._arr.copy()
        _xy[2:] += _xy[:2]  # type: ignore  # x, y, x+w, y+h
        _xy = _xy[[0, 2, 1, 3]]
        return _xy

    @cached_property
    def wh(self):
        if self.kind == "wh":
            return self._arr.copy()
        # self._arr is of the form x1, x2, y1, y2
        _wh = self._arr.copy()
        _wh = _wh[[0, 2, 1, 3]]  # x1, y1, x2, y2
        _wh[2:] -= _wh[:2]  # x, y, x+w, y+h
        return _wh

    @cached_property
    def area(self) -> float:
        return self.wh[2] * self.wh[3]

    def intersection_area(self, other: "BoundingBox") -> float:
        _x1 = max(self.xy[0], other.xy[0])
        _x2 = min(self.xy[1], other.xy[1])
        _y1 = max(self.xy[2], other.xy[2])
        _y2 = min(self.xy[3], other.xy[3])
        return (_x2 - _x1) * (_y2 - _y1)

    # TODO: lru_cache this function: needs a "hash" method
    def iou(self, other: "BoundingBox"):
        intersection_area = self.intersection_area(other)
        union_area = self.area + other.area
        return intersection_area / (union_area - intersection_area)

    @staticmethod
    def from_dict(
        coord_dict: dict[str, float], width: int, height: int
    ) -> "BoundingBox":
        scaler = max(width, height)
        _arr = (
            coord_dict["x1"] / scaler,
            coord_dict["x2"] / scaler,
            coord_dict["y1"] / scaler,
            coord_dict["y2"] / scaler,
        )
        return BoundingBox(coordinates=_arr, kind="xy")

    @staticmethod
    def from_normalized_dict(coord_dict: dict[str, float]) -> "BoundingBox":
        return BoundingBox.from_dict(coord_dict, width=1, height=1)


class Frame:
    def __init__(
        self,
        unique_id: str,
        object_types: list[Obj_Type],
        width: int,
        height: int,
        file_path=None,
    ) -> None:
        self.id = unique_id
        self.object_types = object_types
        self.file_path = file_path
        self.width = width
        self.height = height


class Data_Point:
    "Represents a single data point for an individual object from a specific node."

    def __init__(self, arr, node: Node, obj_name: str, bbox_kind="xy") -> None:
        self._arr = arr
        self.bbox = BoundingBox(arr, kind=bbox_kind)
        self.obj_name = obj_name
        self.node = node
        self.node.register(self)

    @staticmethod
    def from_unnormalized_array(
        arr, node: Node, obj_name: str, width: int, height: int, bbox_kind="xy"
    ) -> "Data_Point":
        _arr = np.array(arr) / max(width, height)
        return Data_Point(_arr, node, obj_name, bbox_kind=bbox_kind)

    @staticmethod
    def from_dict(
        coord_dict, node: Node, obj_name: str, width: int, height: int, bbox_kind="xy"
    ) -> "Data_Point":
        _arr = (coord_dict["x1"], coord_dict["x2"], coord_dict["y1"], coord_dict["y2"])
        return Data_Point.from_unnormalized_array(
            _arr, node, obj_name, width, height, bbox_kind=bbox_kind
        )


class Data_Set:
    def __init__(
        self,
        points: list[Data_Point],
        object_type: Obj_Type,
        nodes: list[Node],
        frame: Frame,
    ):
        self.points = points
        self.object_type = object_type
        self.nodes = nodes
        self.frame = frame

    @cached_property
    def array(self):
        return np.array([c for point in self.points for c in point.bbox.xy]).reshape(
            len(self.nodes) * self.object_type.count, 4
        )

    @cached_property
    def node_labels(self):
        return np.array([point.node.name for point in self.points]).reshape(
            len(self.nodes) * self.object_type.count
        )

    @cached_property
    def unique_node_labels(self):
        return np.unique(self.node_labels)

    @cached_property
    def sigma_array(self):
        return np.array([point.node.sigma for point in self.points]).reshape(
            len(self.nodes) * self.object_type.count
        )

    @lru_cache(maxsize=10)
    def get_points_from_node(self, node_name):
        return self.array[self.node_labels == node_name]
