import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def box_iou(box1, box2):
    x_min_1, x_max_1, y_min_1, y_max_1 = box1.reshape(4)
    x_min_2, x_max_2, y_min_2, y_max_2 = box2.reshape(4)

    x_min = max(x_min_1, x_min_2)
    y_min = max(y_min_1, y_min_2)
    x_max = min(x_max_1, x_max_2)
    y_max = min(y_max_1, y_max_2)

    area_intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    area_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)
    area_union = area_1 + area_2 - area_intersection

    return area_intersection / area_union

def process_row(row):
    "Ravel each row: process each box from a user"
    return [{
        "annotationType": row["annotationType"],
        "objName": row["objName"],
        "userID": row["userID"],
        "confidence": row["confidence"],
        "weight": 1/(1-row["confidence"]),
        "x0": coord["x0"],
        "x1": coord["x1"],
        "y0": coord["y0"],
        "y1": coord["y1"],
        "coordinates": np.array([coord["x0"], coord["x1"], coord["y0"], coord["y1"],]).reshape(4,),
    } for coord in row["coordinates"]]

def merge_lists(_lists):
    return [_i for _list in _lists for _i in _list]


def get_obj_df(_df, objName):
    raw_obj_df = _df[_df["objName"] == objName]    
    flattened_rows = raw_obj_df.apply(process_row, axis=1)
    return pd.DataFrame(merge_lists(flattened_rows))


# affinity matrix
def get_distance(i, j, _obj_df):
    row1 = _obj_df.iloc[i]
    row2 = _obj_df.iloc[j]

    if row1["userID"] == row2["userID"]: # type: ignore
        return 10**2
    _iou = box_iou(row1["coordinates"], row2["coordinates"]) # type: ignore

    if _iou == 0.0:
        return 10

    return 1 - _iou 
        

def get_distance_matrix(_obj_df) -> np.ndarray:
    n_points = len(_obj_df)
    return np.array(
        [
            [get_distance(i, j, _obj_df) for i in range(n_points)] for j in range(n_points)
        ]
    ).reshape(n_points, n_points)


def get_labels(_obj_df):
    distance_matrix = get_distance_matrix(_obj_df)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        linkage="complete",
        distance_threshold=1.0
    )

    return clusterer.fit_predict(distance_matrix)
    

# distance_matrix

def process_object(_df, objName):
    obj_df = get_obj_df(_df, objName)
    # DO NOT INCLUDE THE FOLLOWING IF THE INPUT IS NORMALIZED
    obj_df[["x0", "x1"]] /= 1920
    obj_df[["y0", "y1"]] /= 1080
    obj_df["coordinates"] = obj_df.apply(lambda row: np.multiply(row.coordinates, [1/1920, 1/1920, 1/1080, 1/1080]), axis=1)
    
    # Labels
    obj_labels = get_labels(obj_df)
    return obj_labels