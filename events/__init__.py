#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modulo de eventos do sistema de backtesting.

Este módulo define as classes base e específicas para os diferentes tipos de eventos
que podem ocorrer durante a execução de um backtest, como eventos de mercado,
sinais de trading, ordens e preenchimentos de ordens.

Classes:
    Event: Classe base para todos os eventos.
    MarketEvent: Representa um novo tick ou barra de dados de mercado.
    SignalEvent: Representa um sinal de trading gerado por uma estratégia.
    OrderEvent: Representa uma ordem de trading enviada ao sistema de execução.
    FillEvent: Representa uma ordem que foi preenchida (total ou parcialmente).
    NewsEvent: Representa um evento de notícia.
    RebalanceEvent: Representa um evento de rebalanceamento de portfólio.
    VolatilityEvent: Representa um evento de volatilidade extrema.

Enums:
    EventType: Enumeração dos tipos de evento (MARKET, SIGNAL, ORDER, FILL, NEWS, REBALANCE, VOLATILITY).
    Direction: Enumeração da direção de um sinal ou ordem (BUY, SELL, HOLD).
    OrderType: Enumeração do tipo de ordem (MARKET, LIMIT, STOP, STOP_LIMIT).
"""


# Importa a classe base
from .base import Event
from .fill_event import FillEvent
# Importa as classes de eventos específicos
from .market_event import MarketEvent
from .news_event import NewsEvent
from .order_event import OrderEvent
from .rebalance_event import RebalanceEvent
from .signal_event import SignalEvent
# Importa os Enums
from .types import Direction, EventType, OrderType
from .volatility_event import VolatilityEvent

# Controla o que é exposto com __all__
__all__ = [
    "EventType",
    "Direction",
    "OrderType",
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "NewsEvent",
    "RebalanceEvent",
    "VolatilityEvent",
]
