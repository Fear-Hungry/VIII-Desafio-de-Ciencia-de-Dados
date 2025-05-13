from enum import Enum

class DataType(Enum):
    """Enumeração para tipos de dados financeiros."""
    OHLCV = "ohlcv"  # Open, High, Low, Close, Volume
    TRADES = "trades"
    ORDER_BOOK = "order_book"
    NEWS = "news"
    # Adicione outros tipos de dados conforme necessário

    def __str__(self):
        return self.value

class DataFrequency(Enum):
    """Enumeração para frequências de dados."""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    # Adicione outras frequências conforme necessário

    def __str__(self):
        return self.value

class DataSourceType(Enum):
    """Enumeração para tipos de fontes de dados."""
    CSV = "csv"
    PARQUET = "parquet"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    # Adicione outras fontes conforme necessário

    def __str__(self):
        return self.value

# Você pode adicionar outras classes ou tipos de dados relevantes aqui
# Por exemplo, um Dataclass para configurar uma fonte de dados específica:
#
# from dataclasses import dataclass
# from typing import Optional, Dict, Any
#
# @dataclass
# class DataSourceConfig:
#     source_type: DataSourceType
#     path: Optional[str] = None  # Para CSV, Parquet
#     connection_details: Optional[Dict[str, Any]] = None # Para Database, API
#     frequency: Optional[DataFrequency] = None
#     data_type: Optional[DataType] = None
#     # Adicione outros campos de configuração
#
#     def __post_init__(self):
#         if self.source_type in [DataSourceType.CSV, DataSourceType.PARQUET] and not self.path:
#             raise ValueError(f"'path' é obrigatório para DataSourceType {self.source_type.name}")
#         if self.source_type in [DataSourceType.DATABASE, DataSourceType.API] and not self.connection_details:
#             raise ValueError(f"'connection_details' é obrigatório para DataSourceType {self.source_type.name}")
