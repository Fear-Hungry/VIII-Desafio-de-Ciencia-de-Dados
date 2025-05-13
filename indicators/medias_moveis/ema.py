import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class EMAIndicator(Indicator):
    """Calcula a Média Móvel Exponencial (EMA).

    A EMA é uma média móvel que dá mais peso aos preços mais recentes,
    tornando-a mais reativa a mudanças recentes de preço do que a SMA.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador EMA com sua configuração."""
        if config.type != IndicatorType.EMA:
            raise ValueError(
                f"Configuração inválida para EMAIndicator. Tipo esperado: EMA, recebido: {config.type}"
            )
        super().__init__(config)
        self.period = int(self.config.params[0])
        # TODO: Permitir configurar a coluna de preço (ex: 'Close', 'Open') via config?
        self.price_column = "close"  # Assume 'close' por padrão

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula a EMA usando Polars.

        Args:
            data: DataFrame Polars com pelo menos a coluna de preço (padrão: 'Close').
                  Deve estar ordenado por data.

        Returns:
            DataFrame Polars com a coluna 'ema_{period}'.
        """

        if self.price_column not in data.columns:
            raise ValueError(
                f"DataFrame de entrada precisa conter a coluna: '{self.price_column}'"
            )

        period = self.period
        alpha = 2.0 / (period + 1.0)
        output_col_name = self.column_name  # Vem da config (ex: ema_14)

        # Calcula a EMA usando ewm_mean
        # adjust=False é mais comum em finanças, focando na recursividade
        # min_periods garante NaNs no início
        ema_series = data.select(
            pl.col(self.price_column)
            .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
            .alias(output_col_name)
        )

        # Se precisar manter a coluna de data/índice original:
        if data.columns:
            result_df = pl.concat(
                [data.select(data.columns[0]), ema_series], how="horizontal"
            )
            # Renomeia a coluna de índice/tempo se necessário (opcional)
            # result_df = result_df.rename({data.columns[0]: "Index"})
        else:
            result_df = ema_series

        return result_df
