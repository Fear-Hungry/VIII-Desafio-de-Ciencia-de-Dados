"""
Tipos de Enumeração para o sistema de eventos.
"""

from enum import Enum, auto


class EventType(Enum):
    """Enumeração dos tipos de evento do sistema"""

    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    NEWS = auto()
    REBALANCE = auto()
    VOLATILITY = auto()


class Direction(Enum):
    """Direção de uma ordem ou sinal"""

    BUY = auto()
    SELL = auto()
    HOLD = auto()


class OrderType(Enum):
    """Tipo de ordem de trading"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
