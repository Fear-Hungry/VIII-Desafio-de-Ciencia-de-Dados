#from backtesting.logger import get_logger

from .ichimoku_cloud import IchimokuCloudIndicator

#logger = get_logger(__name__, log_file="indicators.log")
#logger.debug("Módulo tendencia importado.")

__all__ = [
    "IchimokuCloudIndicator",
]
