import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class ATRIndicator(Indicator):
    """Calcula o Average True Range (ATR)."""

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador ATR."""
        if config.type != IndicatorType.ATR:  # Assume que existe um IndicatorType.ATR
            raise ValueError(
                f"Configuração inválida para ATRIndicator. Tipo esperado: ATR, recebido: {config.type}"
            )
        super().__init__(config)
        # Assume que o período está no primeiro parâmetro
        if not config.params or not isinstance(config.params[0], (int, float)):
            raise ValueError("Parâmetro 'period' inválido ou ausente para ATR.")
        self.period = int(config.params[0])

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o ATR usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High', 'Low', 'Close'.

        Returns:
            DataFrame Polars com a coluna 'ATR_{period}'.
        """
        required_cols = ["high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        period = self.period
        # Usar alpha = 1 / period para EWM, similar ao ADX (Wilder's smoothing)
        alpha = 1.0 / period

        data_with_tr = data.with_columns(
            [
                pl.col("high").alias("high"),
                pl.col("low").alias("low"),
                pl.col("close").alias("close"),
                # Adiciona coluna 'close_prev'
                pl.col("close").shift(1).alias("close_prev"),
            ]
        ).with_columns(
            pl.max_horizontal(  # Calcula o True Range (TR)
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close_prev")).abs(),
                (pl.col("low") - pl.col("close_prev")).abs(),
            ).alias("tr")
        )

        # Calcula o ATR usando Exponential Moving Average (EWM)
        atr_series = data_with_tr.select(
            pl.col("tr")
            .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
            .alias(f"ATR_{period}")
        )

        # Adiciona a coluna ATR ao DataFrame original (ou retorna só a coluna ATR)
        # Para consistência com ADX, vamos retornar um DF com a coluna calculada
        # e a primeira coluna original (assumindo ser data/índice)
        result_df = pl.concat(
            [data.select(data.columns[0]), atr_series], how="horizontal"
        )

        # Preencher NaNs iniciais da EWM (como em ADX)
        final_df = result_df.with_columns(
            # Ou forward? Testar. ADX usa backward.
            pl.col(f"ATR_{period}").fill_null(strategy="backward")
        )

        return final_df  # Retorna só [index, ATR_period]
