from typing import List, Union
import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class MACDIndicator(Indicator):
    """
    Moving Average Convergence Divergence (MACD).

    MACD é um indicador de tendência que mostra a relação entre duas médias móveis
    exponenciais (EMAs) dos preços de um ativo.

    Exemplos:
        >>> # Criar um indicador MACD com parâmetros padrão (12,26,9)
        >>> macd = MACDIndicator(IndicatorConfig(type=IndicatorType.MACD, params=[12, 26, 9]))
        >>> # ou diretamente com os parâmetros
        >>> macd = MACDIndicator([12, 26, 9])
        >>> # Calcular o indicador para um DataFrame
        >>> df_com_macd = macd.calculate(df)
    """

    def __init__(self, config: Union[IndicatorConfig, List[int]]):
        """
        Inicializa o indicador MACD.

        Args:
            config: Configuração do indicador (IndicatorConfig) ou diretamente
                   uma lista com [período_curto, período_longo, sinal].
        """
        if isinstance(config, list):
            # Se for uma lista, assume que são os parâmetros [fast, slow, signal]
            if len(config) != 3:
                raise ValueError("MACD precisa de 3 parâmetros: [fast_period, slow_period, signal_period]")
            self.fast_period, self.slow_period, self.signal_period = config
            # Cria uma configuração padrão com os parâmetros fornecidos
            self.config = IndicatorConfig(
                type=IndicatorType.MACD,
                params=config
            )
        else:
            # Assume que é um IndicatorConfig
            self.config = config
            params = self.config.params
            if len(params) != 3:
                params = [12, 26, 9]  # valores padrão
            self.fast_period, self.slow_period, self.signal_period = params

        # Nomes das colunas de saída
        self.macd_line_col = "macd_line"
        self.signal_line_col = "macd_signal"
        self.histogram_col = "macd_hist"

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o indicador MACD para o DataFrame fornecido.

        Args:
            df: DataFrame com dados OHLCV.

        Returns:
            DataFrame com as colunas do indicador MACD adicionadas.
        """
        if "close" not in df.columns:
            raise ValueError("A coluna 'close' é necessária para calcular o MACD")

        # Calcular EMAs - Alphas
        fast_alpha = 2 / (self.fast_period + 1)
        slow_alpha = 2 / (self.slow_period + 1)
        signal_alpha = 2 / (self.signal_period + 1)

        # Definir as expressões para cálculo
        fast_ema_expr = pl.col("close").ewm_mean(alpha=fast_alpha)
        slow_ema_expr = pl.col("close").ewm_mean(alpha=slow_alpha)

        # A linha MACD é calculada primeiro
        macd_line_expr = (fast_ema_expr - slow_ema_expr).alias(self.macd_line_col)

        # Calcular o DataFrame com a linha MACD
        # As colunas temporárias _fast_ema e _slow_ema não são estritamente necessárias
        # se macd_line_expr for auto-contida, mas para clareza, podemos fazer assim:
        # df_with_macd_line = df.with_columns(macd_line_expr)
        # Ou, para evitar múltiplas `with_columns` para colunas que dependem umas das outras,
        # podemos usar `select` para criar o contexto e depois `with_columns`.

        df_calculated = df.select([
            pl.col("date"), # Preserva a coluna 'date'
            fast_ema_expr.alias("_fast_ema_temp"), # EMA rápida como coluna temporária
            slow_ema_expr.alias("_slow_ema_temp")  # EMA lenta como coluna temporária
        ]).with_columns([
            (pl.col("_fast_ema_temp") - pl.col("_slow_ema_temp")).alias(self.macd_line_col)
        ]).with_columns([
            pl.col(self.macd_line_col).ewm_mean(alpha=signal_alpha).alias(self.signal_line_col)
        ]).with_columns([
            (pl.col(self.macd_line_col) - pl.col(self.signal_line_col)).alias(self.histogram_col)
        ])

        # Retorna um novo DataFrame apenas com 'date' e as colunas MACD calculadas
        return df_calculated.select([
            "date",
            self.macd_line_col,
            self.signal_line_col,
            self.histogram_col
        ])

    def __str__(self) -> str:
        return f"MACD({self.fast_period},{self.slow_period},{self.signal_period})"
