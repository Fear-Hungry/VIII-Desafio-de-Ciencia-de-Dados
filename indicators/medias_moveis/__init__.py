"""
Subpacote de Indicadores de Médias Móveis.

Este subpacote contém implementações de indicadores baseados em médias móveis,
incluindo SMA, EMA, MACD e ADX.
"""

#from backtesting.logger import get_logger

from .adx import ADXIndicator
from .ema import EMAIndicator
from .macd import MACDIndicator
# Importar indicadores específicos
from .sma import SMAIndicator

#logger = get_logger(__name__, log_file="indicators.log")
#logger.debug("Módulo medias_moveis importado.")

# Controla o que é exposto com __all__
__all__ = ["SMAIndicator", "EMAIndicator", "MACDIndicator", "ADXIndicator"]
