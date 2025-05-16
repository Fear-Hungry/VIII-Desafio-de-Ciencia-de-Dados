import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class BollingerBandsIndicator(Indicator):
    """Calcula as Bandas de Bollinger."""

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador Bollinger Bands."""
        # Assume IndicatorType.BB
        if config.type != IndicatorType.BB:
            raise ValueError(
                f"Configuração inválida para BollingerBandsIndicator. Tipo esperado: BB, recebido: {config.type}"
            )
        super().__init__(config)
        # Assume [period, std_dev] em params
        if (
            not config.params
            or len(config.params) < 2
            or not isinstance(config.params[0], (int, float))
            or not isinstance(config.params[1], (int, float))
        ):
            raise ValueError(
                "Parâmetros 'period' ou 'std_dev' inválidos ou ausentes para Bollinger Bands."
            )
        self.period = int(config.params[0])
        self.std_dev = float(config.params[1])

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula as Bandas de Bollinger usando Polars.

        Args:
            data: DataFrame Polars com a coluna 'Close'.

        Returns:
            DataFrame Polars com colunas 'BB_Middle_{period}', 'BB_Upper_{period}', 'BB_Lower_{period}'.
        """
        if "close" not in data.columns:
            raise ValueError("DataFrame de entrada precisa conter a coluna 'close'.")

        period = self.period
        std_dev_multiplier = self.std_dev
        close_col = pl.col("close")

        # Calcular Média Móvel Simples (SMA) e Desvio Padrão Móvel
        middle_band = close_col.rolling_mean(window_size=period, min_periods=period)
        rolling_std = close_col.rolling_std(window_size=period, min_periods=period)

        # Calcular Bandas Superior e Inferior
        upper_band = middle_band + (rolling_std * std_dev_multiplier)
        lower_band = middle_band - (rolling_std * std_dev_multiplier)

        # Criar DataFrame de resultado
        if "date" not in data.columns:
            raise ValueError("Coluna 'date' não encontrada no DataFrame de entrada.")

        result_df = data.with_columns(
            [
                middle_band.alias(f"BB_Middle_{self.period}"),
                upper_band.alias(f"BB_Upper_{self.period}"),
                lower_band.alias(f"BB_Lower_{self.period}"),
            ]
        ).select(
            [
                pl.col("date"),
                pl.col(f"BB_Middle_{self.period}"),
                pl.col(f"BB_Upper_{self.period}"),
                pl.col(f"BB_Lower_{self.period}"),
            ]
        )

        return result_df
