"""
Define o evento de Notícia (NewsEvent).
"""


from .base import Event
from .types import EventType



class NewsEvent(Event):
    """
    Representa um evento de notícia que pode impactar o mercado.
    """

    def __init__(self, timestamp, source, headline, summary):
        """
        Inicializa o NewsEvent.

        Args:
            timestamp: O timestamp do evento.
            source: A fonte da notícia (ex: 'Reuters', 'Bloomberg').
            headline: A manchete da notícia.
            summary: Um resumo da notícia.
        """
        self.type = EventType.NEWS
        self.timestamp = timestamp
        self.source = source
        self.headline = headline
        self.summary = summary

    def __str__(self):
        return (
            f"Type: {self.type.name}, Timestamp: {self.timestamp}, "
            f"Source: {self.source}, Headline: {self.headline}"
        )

    def __repr__(self):
        return str(self)
