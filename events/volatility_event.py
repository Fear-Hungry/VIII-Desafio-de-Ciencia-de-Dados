"""
Define o evento de Volatilidade Extrema (VolatilityEvent).
"""


from .base import Event
from .types import EventType


class VolatilityEvent(Event):
    """
    Representa um evento de volatilidade extrema no mercado.

    Pode ser usado para acionar estratégias de proteção ou pausa nas negociações.
    """

    def __init__(self, timestamp, symbol, volatility_measure):
        """
        Inicializa o VolatilityEvent.

        Args:
            timestamp: O timestamp do evento.
            symbol: O símbolo do ativo que apresentou volatilidade extrema.
            volatility_measure: Uma medida quantitativa da volatilidade (ex: desvio padrão).
        """
        self.type = EventType.VOLATILITY
        self.timestamp = timestamp
        self.symbol = symbol
        self.volatility_measure = volatility_measure

    def __str__(self):
        return (
            f"Type: {self.type.name}, Timestamp: {self.timestamp}, "
            f"Symbol: {self.symbol}, Volatility: {self.volatility_measure}"
        )

    def __repr__(self):
        return str(self)
