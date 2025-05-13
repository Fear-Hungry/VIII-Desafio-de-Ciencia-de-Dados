import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class SMAIndicator(Indicator):
    """Calcula a Média Móvel Simples (SMA).

    A SMA representa o preço médio de um ativo durante um período específico.
    É calculada somando os preços de fechamento recentes e dividindo pelo
    número de períodos.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador SMA com sua configuração."""
        if config.type != IndicatorType.SMA:
            raise ValueError(
                f"Configuração inválida para SMAIndicator. Tipo esperado: SMA, recebido: {config.type}"
            )
        super().__init__(config)
        self.period = int(self.config.params[0])
        self.price_column = "close"  # Assume 'close' por padrão

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula a SMA usando Polars.

        Args:
            data: DataFrame Polars com pelo menos a coluna de preço (padrão: 'Close').
                  Deve estar ordenado por data.

        Returns:
            DataFrame Polars com a coluna 'sma_{period}'.
        """

        if self.price_column not in data.columns:
            raise ValueError(
                f"DataFrame de entrada precisa conter a coluna: '{self.price_column}'"
            )

        output_col_name = self.column_name  # Ex: sma_20

        # Calcula a SMA usando rolling_mean
        # min_periods garante NaNs no início
        sma_series = data.select(
            pl.col(self.price_column)
            .rolling_mean(window_size=self.period, min_periods=self.period)
            .alias(output_col_name)
        )

        # Combina com a coluna de índice/tempo original
        if data.columns:
            result_df = pl.concat(
                [data.select(data.columns[0]), sma_series], how="horizontal"
            )
        else:
            result_df = sma_series

        return result_df
