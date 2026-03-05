from dataclasses import dataclass
from typing import Any


@dataclass
class SearchInfo:
    endpoint: str
    credential: Any
    index_name: str
