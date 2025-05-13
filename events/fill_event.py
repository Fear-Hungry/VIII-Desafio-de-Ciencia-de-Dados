"""
Evento que representa a execução (preenchimento) de uma ordem.
"""

import datetime
from dataclasses import dataclass
from typing import Optional


from .base import Event
from .types import Direction, EventType



@dataclass
class FillEvent(Event):
    """Evento gerado pelo ExecutionHandler que confirma a execução
    de uma ordem com detalhes sobre a transação."""

    symbol: str
    quantity: float
    direction: Direction
    fill_price: float
    commission: float = 0.0
    exchange: str = "SIMULATED"
    order_id: Optional[str] = None  # Geralmente uma string, não int

    # Corrigido o __init__ que estava dentro do __str__ no original
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        fill_price: float,
        commission: float = 0.0,  # Default aqui também
        exchange: str = "SIMULATED",
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
    ):
        # Chamada super correta e indentação corrigida
        super().__init__(EventType.FILL, timestamp)
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission
        self.exchange = exchange
        self.order_id = order_id

    def __str__(self) -> str:
        # Formato esperado pelos testes
        return (
            f"Fill: Timestamp={self.timestamp} Symbol={self.symbol} "
            f"Direction={self.direction.name} Quantity={self.quantity} "
            f"Price={self.fill_price} Commission={self.commission}"
        )

    def __repr__(self) -> str:
        return str(self)

    def calculate_cost(self) -> float:
        """
        Calcula o valor líquido da transação, considerando a comissão.
        Para compras (BUY), o custo é (quantidade * preço) + comissão.
        Para vendas (SELL), o valor recebido líquido é (quantidade * preço) - comissão.
        Este método retorna o custo para compras e o valor líquido para vendas.

        Returns:
            float: Custo da transação para BUY, valor líquido para SELL.
        """
        if self.direction == Direction.BUY:
            return (self.quantity * self.fill_price) + self.commission
        elif self.direction == Direction.SELL:
            return (self.quantity * self.fill_price) - self.commission
        else:
            # Caso de HOLD ou outro inesperado, retorna 0 ou lança erro?
            # Por enquanto, retornando 0 para evitar quebrar.
            return 0.0
