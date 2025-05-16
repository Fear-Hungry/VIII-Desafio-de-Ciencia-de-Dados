"""
Define o evento de Rebalanceamento (RebalanceEvent).
"""


from .base import Event
from .types import EventType



class RebalanceEvent(Event):
    """
    Representa um evento que sinaliza a necessidade de rebalancear o portfólio.

    Pode ser disparado periodicamente (ex: mensalmente) ou com base em condições de mercado.
    """

    def __init__(self, timestamp):
        """
        Inicializa o RebalanceEvent.

        Args:
            timestamp: O timestamp do evento.
        """
        self.type = EventType.REBALANCE
        self.timestamp = timestamp

    def __str__(self):
        return f"Type: {self.type.name}, Timestamp: {self.timestamp}"

    def __repr__(self):
        return str(self)
