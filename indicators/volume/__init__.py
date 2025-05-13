"""
Subpacote de Indicadores de Volume.

Este subpacote contém implementações de indicadores baseados em volume,
incluindo OBV, MFI e VWAP.
"""

#from backtesting.logger import get_logger

from .mfi import MFIIndicator
# Importar indicadores específicos
from .obv import OBVIndicator
from .vwap import VWAPIndicator

#logger = get_logger(__name__, log_file="indicators.log")
#logger.debug("Módulo volume importado.")

# Controla o que é exposto com __all__
__all__ = ["OBVIndicator", "MFIIndicator", "VWAPIndicator"]
