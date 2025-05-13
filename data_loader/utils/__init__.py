"""
Utilitários para o Data Loader.

Este pacote contém módulos auxiliares para tarefas como:
- Reamostragem de dados (resampling)
- Ajustes de dados (splits, dividendos) - Placeholders
- Outras operações comuns de pré-processamento.
"""

from .adjustments import apply_dividends, apply_splits
# Importa funções diretamente para facilitar o acesso
from .resampling import align_multiple_timeframes, get_timeframe, resample_ohlc

__all__ = [
    "resample_ohlc",
    "get_timeframe",
    "align_multiple_timeframes",
    "apply_splits",  # Placeholder
    "apply_dividends",  # Placeholder
]
