import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class RSIIndicator(Indicator):
    """Calcula o Índice de Força Relativa (Relative Strength Index - RSI).

    O RSI é um oscilador de momentum que mede a velocidade e a mudança
    dos movimentos de preços. Ele varia de 0 a 100 e é comumente usado
    para identificar condições de sobrecompra ou sobrevenda.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador RSI com sua configuração."""
        if config.type != IndicatorType.RSI:
            raise ValueError(
                f"Configuração inválida para RSIIndicator. Tipo esperado: RSI, recebido: {config.type}"
            )
        super().__init__(config)
        self.period = int(self.config.params[0])
        self.price_column = "close"  # Assume 'close' por padrão

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o RSI usando Polars, aplicando Wilder's smoothing (EMA com alpha=1/period).

        Args:
            data: DataFrame Polars com pelo menos a coluna de preço (padrão: 'Close').
                  Deve estar ordenado por data.

        Returns:
            DataFrame Polars com a coluna 'rsi_{period}'.
        """

        if self.price_column not in data.columns:
            raise ValueError(
                f"DataFrame de entrada precisa conter a coluna: '{self.price_column}'"
            )

        output_col_name = self.column_name  # Ex: rsi_14
        price_col = pl.col(self.price_column)
        period = self.period
        alpha = 1.0 / period

        # Calcula a diferença de preço
        delta = price_col.diff()

        # Separa ganhos e perdas
        gain_expr = pl.when(delta > 0).then(delta).otherwise(pl.lit(0))
        loss_expr = pl.when(delta < 0).then(-delta).otherwise(pl.lit(0)) # Se delta < 0, -delta é positivo (perda)

        # Calcula a média suavizada (Wilder's = EMA com alpha=1/period) dos ganhos e perdas
        # Precisamos calcular sobre o DataFrame para ter as colunas gain/loss
        rsi_calcs = data.with_columns([
            gain_expr.alias("gain"),
            loss_expr.alias("loss")
        ]).select([
            pl.col("gain")
            .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
            .alias("avg_gain"),
            pl.col("loss")
            .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
            .alias("avg_loss"),
        ])

        # Calcula RS e RSI
        rsi_series = rsi_calcs.select(
            pl.when(pl.col("avg_loss") == 0)
            .then(100.0)  # Se avg_loss é 0, RSI é 100 (ou indefinido, 100 é comum)
            .otherwise(
                100.0 - (100.0 / (1.0 + pl.col("avg_gain") / pl.col("avg_loss")))
            )
            .alias(output_col_name)
        )

        # Preencher NaNs iniciais da EWM - fill com 50.0 é uma opção comum para RSI, ou bfill
        rsi_series = rsi_series.with_columns(
            pl.col(output_col_name).fill_null(strategy="backward")
        )  # Ou .fill_null(50.0)

        # Combina com a coluna de índice/tempo original
        if data.columns:
            result_df = pl.concat(
                [data.select("date"), rsi_series], how="horizontal"
            )
        else:
            result_df = rsi_series

        return result_df
