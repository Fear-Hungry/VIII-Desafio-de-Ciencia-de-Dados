import datetime
from abc import ABC, abstractmethod
from typing import List, Optional

import polars as pl  # Importado polars em vez de pandas

from events import MarketEvent

"""
**Módulo Base do Data Loader (`data_loader.base`)**

Este módulo define a interface abstrata (`DataHandler`) para todos os
manipuladores de dados de mercado dentro do sistema de backtesting.

O propósito é garantir que qualquer fonte de dados (CSV, banco de dados, API, etc.)
seja compatível com a `BacktestingEngine`, fornecendo um conjunto consistente
de métodos para acessar os dados históricos.

Qualquer classe concreta que herde de `DataHandler` **deve** implementar
todos os métodos abstratos definidos aqui.
"""


# Importado MarketEvent do local correto



class DataHandler(ABC):
    """
    **Interface Abstrata para Manipuladores de Dados de Mercado**

    Esta classe define o contrato que todos os manipuladores de dados devem seguir.
    Ela garante que a `BacktestingEngine` possa interagir com diferentes fontes
    de dados de maneira uniforme.

    As implementações concretas (como `CSVDataHandler`) devem herdar desta classe
    e fornecer lógica específica para carregar e fornecer os dados.
    """

    @property
    @abstractmethod
    def symbols(self) -> List[str]:
        """
        **[Abstrato]** Retorna a lista de símbolos (tickers) de ativos que este
        manipulador está gerenciando.

        Returns:
            List[str]: Uma lista contendo os nomes dos símbolos (ex: ['AAPL', 'GOOG']).
        """
        raise NotImplementedError("Subclasses devem implementar symbols()")

    @property
    @abstractmethod
    def continue_backtest(self) -> bool:
        """
        **[Abstrato]** Indica se ainda existem dados históricos a serem processados.

        A `BacktestingEngine` usa esta propriedade para determinar quando parar o loop
        principal do backtest.

        Returns:
            bool: `True` se houver mais barras/ticks de dados disponíveis,
                  `False` caso contrário.
        """
        raise NotImplementedError("Subclasses devem implementar continue_backtest()")

    @abstractmethod
    def get_latest_bars_df(self, symbol: str, N: int = 1) -> Optional[pl.DataFrame]:
        """
        **[Abstrato]** Retorna as N barras de dados mais recentes para um determinado símbolo.

        Args:
            symbol (str): O símbolo do ativo para o qual buscar os dados (ex: 'AAPL').
            N (int): O número de barras recentes a serem retornadas (padrão: 1).

        Returns:
            Optional[pl.DataFrame]: Um DataFrame Polars contendo as N barras mais recentes
                                     (incluindo a atual), ordenadas cronologicamente
                                     (a mais recente no final). As colunas esperadas são
                                     tipicamente 'date' (ou timestamp), 'open', 'high',
                                     'low', 'close', 'volume'.
                                     Retorna `None` se o símbolo não existir ou se não
                                     houver N barras disponíveis ainda.
        """
        raise NotImplementedError("Subclasses devem implementar get_latest_bars_df()")

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> Optional[pl.DataFrame]:
        """
        **[Abstrato]** Retorna as N barras de dados mais recentes para um determinado símbolo.

        Este é um alias para `get_latest_bars_df` e deve retornar um DataFrame Polars.
        Mantido por razões de compatibilidade ou preferência de nomenclatura.

        Args:
            symbol (str): O símbolo do ativo (ex: 'AAPL').
            N (int): O número de barras recentes a serem retornadas (padrão: 1).

        Returns:
            Optional[pl.DataFrame]: Um DataFrame Polars com as N barras mais recentes.
                                     Retorna `None` se o símbolo não for encontrado ou
                                     não houver dados suficientes.
        """
        raise NotImplementedError("Subclasses devem implementar get_latest_bars()")

    @abstractmethod
    def get_latest_bar_value(self, symbol: str, val_type: str) -> Optional[float]:
        """
        **[Abstrato]** Retorna um valor específico da barra mais recente para um símbolo.

        Útil para obter rapidamente o preço de fechamento mais recente ou outro valor
        individual sem precisar buscar um DataFrame inteiro.

        Args:
            symbol (str): O símbolo do ativo (ex: 'AAPL').
            val_type (str): O nome da coluna desejada (ex: 'close', 'open', 'volume').
                              Espera-se que seja insensível a maiúsculas/minúsculas.

        Returns:
            Optional[float]: O valor numérico da coluna `val_type` para a barra mais
                             recente do `symbol`. Retorna `None` se o símbolo, a coluna
                             ou os dados não existirem.
        """
        raise NotImplementedError("Subclasses devem implementar get_latest_bar_value()")

    @abstractmethod
    def get_latest_bar_datetime(self, symbol: str) -> Optional[datetime.datetime]:
        """
        **[Abstrato]** Retorna o timestamp da barra mais recente disponível para um símbolo.

        Args:
            symbol (str): O símbolo do ativo (ex: 'AAPL').

        Returns:
            Optional[datetime.datetime]: O timestamp (data e hora) da última barra
                                        processada para o `symbol`. Retorna `None` se
                                        não houver dados para o símbolo.
        """
        raise NotImplementedError(
            "Subclasses devem implementar get_latest_bar_datetime()"
        )

    @abstractmethod
    def update_bars(self) -> Optional[MarketEvent]:
        """
        **[Abstrato]** Avança o estado interno do manipulador de dados para o próximo
        ponto no tempo (próxima barra ou tick) disponível nos dados históricos.

        Este método é o coração do loop de eventos da engine. Ele deve:
        1. Determinar qual é o próximo timestamp disponível nos dados.
        2. Tornar os dados para esse timestamp (para todos os símbolos relevantes)
           acessíveis através dos métodos `get_latest_*`.
        3. Gerar e retornar um `MarketEvent` contendo o novo timestamp.

        Returns:
            Optional[MarketEvent]: Um evento `MarketEvent` sinalizando que novos dados
                                   estão disponíveis, com o `timestamp` correspondente.
                                   Retorna `None` quando não há mais dados históricos
                                   para processar, sinalizando o fim do backtest.
        """
        raise NotImplementedError("Subclasses devem implementar update_bars()")

    @abstractmethod
    def get_all_bars_between(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> pl.DataFrame:
        """
        **[Abstrato]** Retorna todas as barras disponíveis entre duas datas.

        Args:
            start_date (datetime.datetime): A data de início.
            end_date (datetime.datetime): A data de término.
        """

        raise NotImplementedError("Subclasses devem implementar get_all_bars_between()")

    @abstractmethod
    def get_all_bars_in_period(self, symbol: str, period: str) -> pl.DataFrame:
        raise NotImplementedError(
            "Subclasses devem implementar get_all_bars_in_period()"
        )
