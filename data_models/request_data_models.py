"""
Classes defined here are for validation and deserialization of the request payload.
Use of camelCase required for automatic deserialization of the incoming data
"""


from functools import cached_property
from typing import Iterable, List, Optional, SupportsInt, TypeVar, Union
import numpy as np


# from dataclasses import dataclass, field

from pydantic import Field, conlist, root_validator, validator, NonNegativeInt
from pydantic import BaseModel as PydanticBaseModel
from enum import Enum

# all classes whose name starting with `Raw` meant to represent incoming data and for internal use


class BaseModel(PydanticBaseModel):
    class Config:
        allow_mutation = True
        # arbitrary_types_allowed = True
        keep_untouched = (cached_property,)


class AnnotationType(Enum):
    boundingBox = "boundingBox"
    multiClassification = "multiClassification"
    polygon = "polygon"


class RawObjectCountDatum(BaseModel):
    _annotationType_ = AnnotationType.multiClassification

    __root__: NonNegativeInt = Field(..., example=2)

    @cached_property
    def array(self) -> np.ndarray:
        return np.ndarray([self.__root__], dtype=int)

    @cached_property
    def area(self):
        return None


class RawBoundingBoxDatum(BaseModel):
    _annotationType_ = AnnotationType.boundingBox

    x1: int = Field(..., ge=0, example=235)
    x2: int = Field(..., ge=0, example=340)
    y1: int = Field(..., ge=0, example=450)
    y2: int = Field(..., ge=0, example=543)

    @root_validator
    def validate_x2_is_greater_than_x1(cls, values):
        if (values["x1"] >= values["x2"]) or (values["y1"] >= values["y2"]):
            raise ValueError(f"Not a valid bounding box: {values=}")
        return values

    @cached_property
    def array(self) -> np.ndarray:
        return np.array((self.x1, self.y1, self.x2, self.y2), dtype=int)


class RawPolygonDatum(BaseModel):
    _annotationType_ = AnnotationType.polygon

    __root__: List[NonNegativeInt] = Field(..., min_items=2)

    @root_validator
    def validate_coordinates_are_two_tuples(cls, values):
        _items = values["__root__"]
        if len(_items) % 2:
            raise ValueError(
                f"Polygon data is expected to be of the form `x0, y0, x1, y1, x2, y2, ...`; but this sequence of length {len(_items)} cannot be parsed into valid (x, y) pairs: {str(_items)[:50]}"
            )
        return values

    @cached_property
    def array(self) -> np.ndarray:
        return np.array(self.__root__, dtype=int)


RawAnnotationsData = Union[List[RawBoundingBoxDatum], List[RawPolygonDatum]]


class ImageWidthHeight(BaseModel):
    imageWidth: int = Field(..., gt=0, example=1920)
    imageHeight: int = Field(..., gt=0, example=1080)


class DailyTimeStamp(BaseModel):
    _seconds: int = Field(..., example=1648724146)
    _nanoseconds: int = Field(..., example=616110000)


class RawAnnotation(BaseModel):
    """
    Exact python representation of the data model employed in firebase:
    Encapsulates not only the numerical data for entities of the same type from a single user but also
    includes userID, objectCount, image data, etc.
    """

    imageWidthHeight: ImageWidthHeight
    annotationType: AnnotationType
    imageID: Optional[str] = Field("", example="nxv3D8cDK93RPhgsoeGW")
    isGolden: Optional[bool] = Field(None)
    objectCount: int = Field(..., ge=0, example=3)
    userID: str = Field(..., example="sMTqxxXs2gM8yzYuR42EmStUdeM2")
    dailyTimeStamp: DailyTimeStamp = Field(...)
    annotation: Union[RawAnnotationsData, RawObjectCountDatum] = Field(...)
    # TODO: set default confidence level here
    projectBasedConfidence: float = Field(0.6, ge=0.0, le=1.0, example=0.832421)
    projectID: Optional[str] = Field("", example="g8LxKNp8DjqbqIlIZYGG")

    @validator("annotation")
    def validate_each_annotation_is_of_the_same_type(cls, annotation, values):
        ann_type = values["annotationType"]
        if ann_type == AnnotationType.multiClassification:
            if annotation._annotationType_ != ann_type:
                raise ValueError("values")  # TODO: write a decent error msg
            return annotation

        for annt in annotation:
            if ann_type != annt._annotationType_:
                raise ValueError("values")
        return annotation

    @validator("objectCount")
    def validate_object_count_greater_than_zero(cls, obj_count: int, values: dict):
        _annotation_type: AnnotationType = values["annotationType"]
        if _annotation_type != AnnotationType.multiClassification and obj_count == 0:
            raise ValueError(
                f"Cannot validate '{_annotation_type}' data when object count is 0."
            )
        return obj_count

    @validator("annotation")
    def validate_annotation_has_object_count_many_elements(cls, annotation, values):
        if values["annotationType"] == AnnotationType.multiClassification:
            values["objectCount"] = 0
            return annotation
        if len(annotation) != values["objectCount"]:
            raise ValueError(
                f"There must be exactly {values['objectCount']} many elements.\n{values=}"
            )
        return annotation

    @cached_property
    def nonnormalized_array(self):
        """
        Returns an np.ndarray of shape objectCount x data_dim
        where data_dim is the inherent dimension of the annotation data (ie. 4 when annotationType==boundingBox).
        If annotationType==multiClassification, return array[int].
        """
        if self.annotationType == AnnotationType.multiClassification:
            assert isinstance(self.annotation, RawObjectCountDatum)
            return np.array((self.annotation.__root__), dtype=int)

        assert not isinstance(self.annotation, RawObjectCountDatum)
        return np.array([annt.array for annt in self.annotation], ndmin=2, dtype=int)

    @cached_property
    def array(self) -> np.ndarray:
        if self.annotationType == AnnotationType.multiClassification:
            return self.nonnormalized_array
        normalizing_factor = max(
            self.imageWidthHeight.imageWidth, self.imageWidthHeight.imageHeight
        )
        return self.nonnormalized_array / normalizing_factor


class ValidationRequestPayload(BaseModel):
    """
    Example usage: ValidationRequestPayload.parse_raw(BBOX_INPUT_STR, content_type="json")
    """

    annotations: List[RawAnnotation] = Field(..., min_items=1)
    objectPath: str = Field(...)

    @validator("annotations", each_item=True)
    def validate_is_not_golden(cls, raw_annotation):
        if raw_annotation.isGolden is True:
            raise ValueError("Try to validate when 'isGolden' flag is set to true.")
        return raw_annotation

    @validator("annotations")
    def validate_annotations_come_from_different_users(
        cls, annotations: list[RawAnnotation]
    ):
        user_ids = [annt.userID for annt in annotations]
        # all user_ids unique?
        assert len(set(user_ids)) == len(
            user_ids
        ), f"RETRY: annotations originate from the same users! {user_ids:}"
        return annotations
