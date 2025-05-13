"""
Módulo de Indicadores Técnicos para Análise Financeira.

Este módulo fornece uma variedade de indicadores técnicos amplamente utilizados
em análise financeira e desenvolvimento de estratégias de trading.

Classes principais:
    - Indicator: Classe base abstrata para todos os indicadores
    - IndicatorConfig: Configuração para indicadores
    - IndicatorType: Enumeração dos tipos de indicadores suportados

O pacote está organizado em subpacotes por categoria:
    - medias_moveis: Indicadores baseados em médias móveis (SMA, EMA, MACD, ADX)
    - momento: Indicadores de momento (RSI, ROC, Stochastic, CCI)
    - volatilidade: Indicadores de volatilidade (Bollinger Bands, ATR, Donchian Channels)
    - volume: Indicadores baseados em volume (OBV, MFI, VWAP)
    - niveis: Indicadores de níveis de suporte/resistência (Fibonacci Retracement)
    - tendencia: Indicadores de tendência (Ichimoku Cloud)
"""

# Importar subpacotes
# Nota: Cada subpacote deve ter seu próprio __init__.py configurado corretamente
from . import medias_moveis, momento, niveis, tendencia, volatilidade, volume
from .base import Indicator
# Importar tipos e classes base
from .types import IndicatorConfig, IndicatorType

# Controla o que é exposto com __all__
__all__ = [
    # Classes base
    "Indicator",
    "IndicatorConfig",
    "IndicatorType",
    # Subpacotes
    "medias_moveis",
    "momento",
    "volatilidade",
    "volume",
    "niveis",
    "tendencia",
]

import logging
logger = logging.getLogger(__name__)
# Se você quiser configurar um arquivo de log básico rapidamente aqui:
# logging.basicConfig(filename='indicators.log', level=logging.INFO) # Ou logging.DEBUG, etc.
# No entanto, é geralmente melhor configurar o logging no ponto de entrada da sua aplicação.
