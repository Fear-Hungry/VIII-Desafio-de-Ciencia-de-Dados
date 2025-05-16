"""
Classe base para todos os eventos do sistema.
"""

import datetime
from abc import ABC
from dataclasses import dataclass
from typing import Optional


from .types import EventType



@dataclass
class Event(ABC):
    """Classe base abstrata para todos os eventos"""

    type: EventType
    timestamp: Optional[datetime.datetime]
