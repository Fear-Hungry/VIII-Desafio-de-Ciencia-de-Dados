import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class ADXIndicator(Indicator):
    """Calcula o Average Directional Index (ADX), Plus Directional Indicator (+DI)
    e Minus Directional Indicator (-DI).

    O ADX mede a força de uma tendência (independente da direção),
    enquanto +DI e -DI medem a força direcional positiva e negativa,
    respectivamente.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador ADX com sua configuração."""
        if config.type != IndicatorType.ADX:
            raise ValueError(
                f"Configuração inválida para ADXIndicator. Tipo esperado: ADX, recebido: {config.type}"
            )
        super().__init__(config)
        self.period = int(self.config.params[0])

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o ADX, +DI e -DI usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High', 'Low', 'Close'. Deve estar ordenado por data.

        Returns:
            DataFrame Polars com colunas 'ADX_{period}', '+DI_{period}', '-DI_{period}'.
        """

        required_cols = ["high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        period = self.period
        alpha = 1.0 / period

        # Calcula TR, +DM, -DM usando expressões Polars
        data_with_shifts = data.with_columns(
            [
                pl.col("high").alias("high"),
                pl.col("low").alias("low"),
                pl.col("close").alias("close"),
                pl.col("close").shift(1).alias("close_prev"),
                pl.col("high").diff().alias("move_up"),
                (-pl.col("low").diff()).alias("move_down"),
            ]
        )

        # 2. Calcular True Range (TR) e suavizar (ATR)
        data_with_tr = data_with_shifts.with_columns(
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close_prev")).abs(),
                (pl.col("low") - pl.col("close_prev")).abs(),
            ).alias("tr")
        ).with_columns(
            pl.col("tr")
            .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
            .alias("atr")
        )

        # 3/4. Calcular +DM, -DM e suavizar
        data_with_dm = data_with_tr.with_columns(
            [
                pl.when(
                    (pl.col("move_up") > pl.col("move_down")) & (pl.col("move_up") > 0)
                )
                .then(pl.col("move_up"))
                .otherwise(0.0)
                .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                .alias("smoothed_plus_dm"),
                pl.when(
                    (pl.col("move_down") > pl.col("move_up"))
                    & (pl.col("move_down") > 0)
                )
                .then(pl.col("move_down"))
                .otherwise(0.0)
                .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                .alias("smoothed_minus_dm"),
            ]
        )

        # 5. Calcular +DI, -DI
        data_with_di = data_with_dm.with_columns(
            [
                (
                    100.0
                    * pl.col("smoothed_plus_dm")
                    / pl.when(pl.col("atr") != 0).then(pl.col("atr")).otherwise(None)
                )
                .fill_null(0.0)
                .alias(f"+DI_{period}"),
                (
                    100.0
                    * pl.col("smoothed_minus_dm")
                    / pl.when(pl.col("atr") != 0).then(pl.col("atr")).otherwise(None)
                )
                .fill_null(0.0)
                .alias(f"-DI_{period}"),
            ]
        )

        # 6/7. Calcular DX e suavizar (ADX)
        di_sum = pl.col(f"+DI_{period}") + pl.col(f"-DI_{period}")
        di_diff = (pl.col(f"+DI_{period}") - pl.col(f"-DI_{period}")).abs()

        data_with_adx = data_with_di.with_columns(
            pl.when(di_sum != 0)
            .then(100.0 * di_diff / di_sum)
            .otherwise(0.0)  # Se soma é 0, DX é 0
            .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
            # .backward_fill() # Preenche NaNs iniciais da EWM
            .alias(f"ADX_{period}")
        ).select(
            [
                # Manter o índice/coluna de data original se existir
                *(data.columns[:1] if data.columns else []),
                pl.col(f"ADX_{period}"),
                pl.col(f"+DI_{period}"),
                pl.col(f"-DI_{period}"),
            ]
        )

        # Preencher NaNs iniciais do ADX que a EWM cria
        # O backward_fill comentado acima não funciona bem em todos os casos
        # Fazemos isso separadamente para mais controle
        final_df = data_with_adx.with_columns(
            pl.col(f"ADX_{period}").fill_null(strategy="backward")
        )

        return final_df
