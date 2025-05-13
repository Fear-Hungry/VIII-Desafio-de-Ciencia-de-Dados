import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class ROCIndicator(Indicator):
    """Calcula a Taxa de Mudança (Rate of Change - ROC).

    O ROC mede a variação percentual no preço entre o período atual e o
    preço de 'n' períodos atrás.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador ROC com sua configuração."""
        if config.type != IndicatorType.ROC:
            raise ValueError(
                f"Configuração inválida para ROCIndicator. Tipo esperado: ROC, recebido: {config.type}"
            )
        super().__init__(config)
        self.period = int(self.config.params[0])
        self.price_column = "close"  # Assume 'close' por padrão

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula a taxa de mudança percentual (ROC) usando Polars.
        ROC = [(PreçoAtual - Preço_n_periodos_atras) / Preço_n_periodos_atras] * 100

        Args:
            data: DataFrame Polars com pelo menos a coluna de preço (padrão: 'Close').
                  Deve estar ordenado por data.

        Returns:
            DataFrame Polars com a coluna 'roc_{period}'.
        """

        if self.price_column not in data.columns:
            raise ValueError(
                f"DataFrame de entrada precisa conter a coluna: '{self.price_column}'"
            )

        output_col_name = self.column_name  # Ex: roc_14
        price_col = pl.col(self.price_column)
        price_shifted = price_col.shift(self.period)

        # Calcula o ROC
        roc_series = data.select(
            pl.when(price_shifted != 0)
            .then((price_col - price_shifted) / price_shifted * 100.0)
            .otherwise(None)  # Retorna null se o preço anterior for 0
            .alias(output_col_name)
        )
        # Preenche NaNs iniciais (do shift e da divisão por zero)
        # O forward fill pode ser apropriado aqui se quisermos propagar o primeiro valor não nulo
        # Mas fill_null(0.0) pode ser mais seguro dependendo da estratégia
        roc_series = roc_series.with_columns(pl.col(output_col_name).fill_null(0.0))

        # Combina com a coluna de índice/tempo original
        if data.columns:
            result_df = pl.concat(
                [data.select(data.columns[0]), roc_series], how="horizontal"
            )
        else:
            result_df = roc_series

        return result_df
