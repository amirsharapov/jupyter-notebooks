import json

from src.json_encoder import JSONEncoder


def prettify(obj):
    return json.dumps(
        obj,
        indent=4,
        cls=JSONEncoder,
        default=str
    )
