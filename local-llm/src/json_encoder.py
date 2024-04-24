import json

from pathlib import Path


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return o.as_posix()
        return super().default(o)
