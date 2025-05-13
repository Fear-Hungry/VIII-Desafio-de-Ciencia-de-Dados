import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class MACDIndicator(Indicator):
    """Calcula o Moving Average Convergence Divergence (MACD).

    O MACD é um indicador de momentum que segue tendências e mostra a relação
    entre duas médias móveis exponenciais dos preços. Ele consiste na linha MACD,
    linha de Sinal (uma EMA da linha MACD) e o Histograma (diferença entre
    MACD e Sinal).
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador MACD com sua configuração."""
        if config.type != IndicatorType.MACD:
            raise ValueError(
                f"Configuração inválida para MACDIndicator. Tipo esperado: MACD, recebido: {config.type}"
            )
        super().__init__(config)
        # Parâmetros: fast, slow, signal
        self.fast_period = int(self.config.params[0])
        self.slow_period = int(self.config.params[1])
        self.signal_period = int(self.config.params[2])
        self.price_column = "close"  # Assume 'close' por padrão

        # Nomes das colunas de saída
        self.param_str = f"{self.fast_period}_{self.slow_period}_{self.signal_period}"
        self.macd_col = f"MACD_{self.param_str}"
        self.signal_col = f"MACDSignal_{self.param_str}"
        self.hist_col = f"MACDHist_{self.param_str}"

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula a linha MACD, linha de Sinal e Histograma.

        Args:
            data: DataFrame Polars com pelo menos a coluna de preço (padrão: 'Close').
                  Deve estar ordenado por data.

        Returns:
            DataFrame Polars com as colunas MACD, Sinal e Histograma.
        """

        if self.price_column not in data.columns:
            raise ValueError(
                f"DataFrame de entrada precisa conter a coluna: '{self.price_column}'"
            )

        # Alphas para as EMAs
        alpha_fast = 2.0 / (self.fast_period + 1.0)
        alpha_slow = 2.0 / (self.slow_period + 1.0)
        alpha_signal = 2.0 / (self.signal_period + 1.0)

        # Calcula EMAs e linha MACD
        data_with_macd = data.with_columns(
            [
                pl.col(self.price_column)
                .ewm_mean(alpha=alpha_fast, adjust=False, min_periods=self.fast_period)
                .alias("fast_ema"),
                pl.col(self.price_column)
                .ewm_mean(alpha=alpha_slow, adjust=False, min_periods=self.slow_period)
                .alias("slow_ema"),
            ]
        ).with_columns((pl.col("fast_ema") - pl.col("slow_ema")).alias(self.macd_col))

        # Calcula linha de Sinal e Histograma
        data_with_signal = data_with_macd.with_columns(
            pl.col(self.macd_col)
            .ewm_mean(alpha=alpha_signal, adjust=False, min_periods=self.signal_period)
            .alias(self.signal_col)
        ).with_columns(
            (pl.col(self.macd_col) - pl.col(self.signal_col)).alias(self.hist_col)
        )

        output_columns = [
            *(data.columns[:1] if data.columns else []),
            self.macd_col,
            self.signal_col,
            self.hist_col,
        ]
        result_df = data_with_signal.select(output_columns)

        return result_df
