"""
Modulo de eventos para o sistema de backtesting.

Este modulo define os diferentes tipos de eventos que podem ocorrer no sistema de backtesting orientado
a eventos.
"""

import datetime
from dataclasses import dataclass
from typing import Optional


from .base import Event
from .types import EventType



@dataclass
class MarketEvent(Event):
    """Evento que sinaliza a chegada de novos dados de mercado
    e aciona a atualização da estrategia"""

    def __init__(self, timestamp: Optional[datetime.datetime] = None):
        super().__init__(EventType.MARKET, timestamp)

    def __str__(self) -> str:
        return f"Type: {self.type.name}, Timestamp: {self.timestamp}"

    def __repr__(self) -> str:
        # É comum fazer o repr retornar algo mais informativo ou igual ao str
        return str(self)
