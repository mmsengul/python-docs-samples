from unittest import TestCase

import numpy as np

from data_models.internal_data_models import (
    Node,
    Obj_Type,
    BoundingBox,
    Frame,
    Data_Point,
    Data_Set,
)

DATA__OK = [
    {
        "annotationType": "boundingBox",
        "objName": "car",
        "coordinates": [
            {"xmin": 100, "ymin": 200, "xmax": 300, "ymax": 400},
            {"xmin": 500, "ymin": 600, "xmax": 700, "ymax": 800},
        ],
        "userID": "UserYZ2",
        "objCount": 2,
        "confidence": 0.9,
    },
    {
        "annotationType": "boundingBox",
        "objName": "car",
        "coordinates": [
            {"xmin": 110, "ymin": 210, "xmax": 310, "ymax": 410},
            {"xmin": 510, "ymin": 610, "xmax": 710, "ymax": 810},
        ],
        "userID": "UserXY2",
        "objCount": 2,
        "confidence": 0.95,
    },
    {
        "annotationType": "boundingBox",
        "objName": "car",
        "coordinates": [
            {"xmin": 130, "ymin": 230, "xmax": 330, "ymax": 430},
            {"xmin": 530, "ymin": 630, "xmax": 730, "ymax": 830},
        ],
        "userID": "UserXZ2",
        "objCount": 2,
        "confidence": 0.98,
    },
]

INPUT_DATA = {
    "frame": {"width": 1920, "height": 1080, "frame_id": "abcd"},
    "annotations": DATA__OK,
}


class Test_BBox_Data_Model(TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_data_model(self):
        _data = INPUT_DATA.copy()
        _obj_types = {
            Obj_Type(obj_name=annt["objName"], _count=annt["objCount"])
            for annt in _data["annotations"]
        }

        _nodes = [
            Node(user_name=annt["userID"], score=annt["confidence"])
            for annt in _data["annotations"]
        ]

        frame = Frame(
            unique_id=_data["frame"]["frame_id"],
            object_types=list(_obj_types),
            width=_data["frame"]["width"],
            height=_data["frame"]["height"],
        )

        _data_points = [
            Data_Point.from_dict(
                coord_dict=coord,
                node=Node(user_name=annt["userID"], score=annt["confidence"]),
                obj_name=annt["objName"],
                width=frame.width,
                height=frame.height,
            )
            for annt in _data["annotations"]
            for coord in annt["coordinates"]
        ]

        _data_set = Data_Set(
            points=_data_points,
            object_type=list(_obj_types)[0],
            nodes=_nodes,
            frame=frame,
        )

        print(_data_set.array)
        print(_data_set.unique_node_labels)
        print(_data_set.sigma_array)
        print(_data_set.get_points_from_node("UserXY2"))
        print(_data_set.get_points_from_node("UserYZ2"))


OBJ_COUNT_INPUT_STR = """{
    "annotations": [
        {
            "projectBasedConfidence": 0.66,
            "dailyTimeStamp": {"_seconds": 1648724146, "_nanoseconds": 616110000},
            "annotationType": "multiClassification",
            "userID": "sMTqxxXs2gM8yzYuR42EmStUdeM2",
            "projectID": "EqEdUgKLPbXbIPzJIN7r",
            "isGolden": false,
            "objectCount": 0,
            "annotation": 3,
            "imageID": "dixJG5UXZTqWjC0zBvDs",
            "imageWidthHeight": {"imageWidth": 500, "imageHeight": 434}
        },
        {
            "projectBasedConfidence": 0.76,
            "dailyTimeStamp": {"_seconds": 1648724146, "_nanoseconds": 616110000},
            "annotationType": "multiClassification",
            "userID": "other_sMTqxxXs2gM8yzYuR42EmStUdeM2",
            "projectID": "EqEdUgKLPbXbIPzJIN7r",
            "isGolden": false,
            "objectCount": 0,
            "annotation": 4,
            "imageID": "dixJG5UXZTqWjC0zBvDs",
            "imageWidthHeight": {"imageWidth": 500, "imageHeight": 434}
        }
    ],
    "objectPath": "Customers/2PooeWFhblfafFiuegYBLHm8o203/projects/EqEdUgKLPbXbIPzJIN7r/dataset/dixJG5UXZTqWjC0zBvDs/objects/WsbGfQv5UvBMvMMaRlBI"
}"""

BBOX_INPUT_STR = """{
    "annotations": [
        {
            "projectBasedConfidence": 0.66,
            "dailyTimeStamp": {"_seconds": 1648724146, "_nanoseconds": 616110000},
            "annotationType": "boundingBox",
            "userID": "sMTqxxXs2gM8yzYuR42EmStUdeM2",
            "projectID": "EqEdUgKLPbXbIPzJIN7r",
            "isGolden": false,
            "objectCount": 2,
            "annotation": [
                {"x1": 130, "y1": 230, "x2": 330, "y2": 430},
                {"x1": 530, "y1": 630, "x2": 730, "y2": 830}
            ],
            "imageID": "dixJG5UXZTqWjC0zBvDs",
            "imageWidthHeight": {"imageWidth": 500, "imageHeight": 434}
        },
        {
            "projectBasedConfidence": 0.76,
            "dailyTimeStamp": {"_seconds": 1648724146, "_nanoseconds": 616110000},
            "annotationType": "boundingBox",
            "userID": "other_sMTqxxXs2gM8yzYuR42EmStUdeM2",
            "projectID": "EqEdUgKLPbXbIPzJIN7r",
            "isGolden": false,
            "objectCount": 2,
            "annotation": [
                {"x1": 180, "y1": 280, "x2": 880, "y2": 480},
                {"x1": 580, "y1": 680, "x2": 780, "y2": 880}
            ],
            "imageID": "dixJG5UXZTqWjC0zBvDs",
            "imageWidthHeight": {"imageWidth": 500, "imageHeight": 434}
        }
    ],
    "objectPath": "Customers/2PooeWFhblfafFiuegYBLHm8o203/projects/EqEdUgKLPbXbIPzJIN7r/dataset/dixJG5UXZTqWjC0zBvDs/objects/WsbGfQv5UvBMvMMaRlBI"
}"""
