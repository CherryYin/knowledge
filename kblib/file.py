from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class File:
    content: Any
    url: Optional[str] = None
    acls: Optional[list[str]] = None
