from dataclasses import dataclass
from typing import Literal

@dataclass
class MessageRequestDTO:
    role: Literal["user", "system"]
    content: str