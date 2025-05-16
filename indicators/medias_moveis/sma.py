from typing import Optional, List, Union

import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class SMAIndicator(Indicator):
    """
    Implementação do indicador SMA (Simple Moving Average/Média Móvel Simples).

    Este indicador calcula a média aritmética dos preços em um período especificado,
    dando igual peso a cada ponto de dados.

    Exemplos:
        >>> # Criar um indicador SMA de 20 períodos
        >>> sma = SMAIndicator(IndicatorConfig(type=IndicatorType.SMA, params=[20]))
        >>> # ou diretamente com o período
        >>> sma = SMAIndicator(20)
        >>> # Calcular o indicador para um DataFrame
        >>> df_com_sma = sma.calculate(df)
    """

    def __init__(self, config: Union[IndicatorConfig, int]):
        """
        Inicializa o indicador SMA.

        Args:
            config: Configuração do indicador (IndicatorConfig) ou diretamente
                   o período como inteiro.
        """
        if isinstance(config, int):
            # Se for só um número, assume que é o período
            self.period = config
            # Cria uma configuração padrão com o período fornecido
            self.config = IndicatorConfig(
                type=IndicatorType.SMA,
                params=[self.period]
            )
        else:
            # Assume que é um IndicatorConfig
            self.config = config
            self.period = self.config.params[0] if self.config.params else 20

        # Nome para a coluna de saída
        self.output_column = f"sma_{self.period}"

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o indicador SMA para o DataFrame fornecido.

        Args:
            df: DataFrame com dados OHLCV.

        Returns:
            DataFrame com a coluna do indicador SMA adicionada.
        """
        if "close" not in df.columns:
            raise ValueError("A coluna 'close' é necessária para calcular o SMA")

        # Calcula SMA com rolling window e nomeia a coluna diretamente na expressão
        sma_expr = pl.col("close").rolling_mean(window_size=self.period).alias(self.output_column)

        # Avalia a expressão no contexto do DataFrame e seleciona as colunas necessárias
        df_with_sma = df.with_columns(sma_expr)

        # Retorna um novo DataFrame contendo apenas o timestamp e o SMA
        return df_with_sma.select(["date", self.output_column])

    def __str__(self) -> str:
        return f"SMA({self.period})"
