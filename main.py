from flask import escape
import xmltodict
import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from process_object import process_object


def _debug(request):
    request_json = request.get_json()
    df = pd.DataFrame(request_json)
    l = []
    for objName in np.unique(df.objName):
        l.append(process_object(df, objName))
    print(l)
    return escape(l.__repr__())


# [START functions_http_xml]

# [END functions_http_xml]
# [START functions_http_xml]

# [END functions_http_xml]
# [START functions_http_xml]


def parse_xml(request):
    """Parses a document of type 'text/xml'
    Args:
        request (flask.Request): The request object.
    Returns:
        The response text, or any set of values that can be turned into a
         Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    data = xmltodict.parse(request.data)
    return json.dumps(data, indent=2)


# [END functions_http_xml]
