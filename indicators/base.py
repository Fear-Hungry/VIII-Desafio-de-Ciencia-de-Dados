from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl  # Importar polars

from .types import IndicatorConfig # Importação direta

import logging
logger = logging.getLogger(__name__)

#from backtesting.logger import get_logger

#logger = get_logger(__name__, log_file="indicators.log")


class Indicator(ABC):
    """Classe base abstrata para todos os indicadores técnicos."""

    def __init__(self, config: IndicatorConfig):
        self.config = config
        # O nome da coluna pode ser derivado da configuração
        # Assume que IndicatorConfig tem column_name
        self.column_name = config.column_name

    @abstractmethod
    def calculate(self, data: pl.DataFrame) -> pl.Series | pl.DataFrame:
        """
        Calcula o indicador nos dados fornecidos.

        Args:
            data: DataFrame do Polars contendo pelo menos os dados de preço
                  (ex: 'Open', 'High', 'Low', 'Close', 'Volume').

        Returns:
            Um Polars Series (para indicadores de linha única como SMA, RSI)
            ou um DataFrame (para indicadores multi-linha como MACD, BBands)
            contendo os valores calculados do indicador. O índice/ordem deve
            corresponder ao DataFrame de entrada.
        """
        pass

    def get_column_name(self) -> str:
        """Retorna o nome da(s) coluna(s) que este indicador gera."""
        # Implementação básica, pode ser sobrescrita se necessário
        return self.column_name

    def __str__(self) -> str:
        return f"Indicator({self.config})"

    def __repr__(self) -> str:
        # Usa repr(self.config) para obter a representação da configuração
        return f"{self.__class__.__name__}(config={repr(self.config)})"
