"""
Subpacote de Indicadores de Volatilidade.

Este subpacote contém implementações de indicadores de volatilidade,
incluindo Bollinger Bands, ATR e Canais de Donchian.
"""

#from backtesting.logger import get_logger

from .atr import ATRIndicator
# Importar indicadores específicos
from .bollinger_bands import BollingerBandsIndicator
from .donchian_channels import DonchianChannelIndicator

#logger = get_logger(__name__, log_file="indicators.log")
#logger.debug("Módulo volatilidade importado.")

# Controla o que é exposto com __all__
__all__ = ["BollingerBandsIndicator", "ATRIndicator", "DonchianChannelIndicator"]
