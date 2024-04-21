import json


def prettify(obj):
    return json.dumps(obj, indent=4)
