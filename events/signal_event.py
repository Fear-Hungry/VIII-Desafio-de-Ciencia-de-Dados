"""
Evento que sinaliza a geração de um sinal de oportunidade de trading.
"""

import datetime
from dataclasses import dataclass
from typing import Optional


from .base import Event
from .types import Direction, EventType



@dataclass
class SignalEvent(Event):
    """Evento que sinaliza a geração de um sinal de oportunidade de trading
    em um ativo específico"""

    symbol: str
    direction: Direction
    strength: float = 1.0  # Força do sinal (0 a 1)

    def __init__(
        self,
        symbol: str,
        direction: Direction,
        strength: float = 1.0,
        timestamp: Optional[datetime.datetime] = None,
    ):
        super().__init__(EventType.SIGNAL, timestamp)
        self.symbol = symbol
        self.direction = direction
        self.strength = strength

    def __str__(self):
        return (
            f"Type: {self.type.name}, Timestamp: {self.timestamp}, "
            f"Symbol: {self.symbol}, Direction: {self.direction.name}, "
            f"Strength: {self.strength}"
        )

    def __repr__(self):
        return str(self)
