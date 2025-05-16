import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class DonchianChannelIndicator(Indicator):
    """Calcula os Canais de Donchian."""

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador Donchian Channels."""
        # Assume IndicatorType.DONCHIAN
        if config.type != IndicatorType.DONCHIAN:
            raise ValueError(
                f"Configuração inválida para DonchianChannelIndicator. Tipo esperado: DONCHIAN, recebido: {config.type}"
            )
        super().__init__(config)
        # Assume que o período está no primeiro parâmetro
        if not config.params or not isinstance(config.params[0], (int, float)):
            raise ValueError(
                "Parâmetro 'period' inválido ou ausente para Donchian Channel."
            )
        self.period = int(config.params[0])

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula os Canais de Donchian usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High' e 'Low'.

        Returns:
            DataFrame Polars com colunas 'DC_Upper_{period}', 'DC_Lower_{period}', 'DC_Middle_{period}'.
        """
        required_cols = ["high", "low"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        period = self.period

        # Calcular Máxima e Mínima Móveis
        upper_channel = pl.col("high").rolling_max(
            window_size=period, min_periods=period
        )
        lower_channel = pl.col("low").rolling_min(
            window_size=period, min_periods=period
        )

        # Calcular Canal Médio
        middle_channel = (upper_channel + lower_channel) / 2

        # Criar DataFrame de resultado
        if "date" not in data.columns:
            raise ValueError("Coluna 'date' não encontrada no DataFrame de entrada.")

        result_df = data.with_columns(
            [
                upper_channel.alias(f"DC_Upper_{self.period}"),
                lower_channel.alias(f"DC_Lower_{self.period}"),
                middle_channel.alias(f"DC_Middle_{self.period}"),
            ]
        ).select(
            [
                pl.col("date"),
                pl.col(f"DC_Upper_{self.period}"),
                pl.col(f"DC_Lower_{self.period}"),
                pl.col(f"DC_Middle_{self.period}"),
            ]
        )

        return result_df
