"""
Evento que sinaliza a criação de uma ordem de negociação.
"""

import datetime
from dataclasses import dataclass
from typing import Optional


from .base import Event
from .types import Direction, EventType, OrderType



@dataclass
class OrderEvent(Event):
    """Evento que sinaliza a geração de uma ordem de trading
    para um ativo específico"""

    symbol: str
    order_type: OrderType
    direction: Direction
    quantity: float
    order_id: Optional[str] = None
    # Necessário para ordens LIMIT e STOP_LIMIT
    price: Optional[float] = None
    # Necessário para ordens STOP e STOP_LIMIT
    stop_price: Optional[float] = None

    def __init__(
        self,
        symbol: str,  # Adicionado o parâmetro symbol que estava faltando
        order_type: OrderType,
        direction: Direction,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime.datetime] = None,
        order_id: Optional[str] = None,
    ):
        super().__init__(EventType.ORDER, timestamp)
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.order_id = order_id

        # Validação básica de preços para tipos de ordem
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Preço é obrigatório para ordens LIMIT.")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop price é obrigatório para ordens STOP.")
        if self.order_type == OrderType.STOP_LIMIT and (
            self.price is None or self.stop_price is None
        ):
            raise ValueError(
                "Preço e stop price são obrigatórios para ordens STOP_LIMIT."
            )

    def __str__(self) -> str:
        """Retorna a representação em string do evento de ordem."""
        # Monta a string base
        base = (
            f"Order: Timestamp={self.timestamp} Symbol={self.symbol} "
            f"Type={self.order_type.name} Direction={self.direction.name} "
            f"Quantity={self.quantity}"
        )

        if self.order_id:
            base += f" OrderID={self.order_id}"

        # Adiciona preço e stop price se existirem
        if self.price is not None:
            base += f" Price={self.price}"
        if self.stop_price is not None:
            base += f" StopPrice={self.stop_price}"

        return base

    def __repr__(self) -> str:
        return str(self)
