import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class MFIIndicator(Indicator):
    """Calcula o Money Flow Index (MFI)."""

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador MFI."""
        if config.type != IndicatorType.MFI:  # Assume IndicatorType.MFI
            raise ValueError(
                f"Configuração inválida para MFIIndicator. Tipo esperado: MFI, recebido: {config.type}"
            )
        super().__init__(config)
        if not config.params or not isinstance(config.params[0], (int, float)):
            raise ValueError("Parâmetro 'period' inválido ou ausente para MFI.")
        self.period = int(config.params[0])

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o MFI usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High', 'Low', 'Close', 'Volume'.

        Returns:
            DataFrame Polars com a coluna 'MFI_{period}'.
        """
        required_cols = ["high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        period = self.period

        # 1. Calcular Typical Price (TP)
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3

        # 2. Calcular Raw Money Flow (RMF)
        rmf = tp * pl.col("volume")

        # 3. Calcular Positive e Negative Money Flow (PMF, NMF)
        #    Comparar TP atual com o TP anterior
        data_with_flows = (
            data.with_columns([tp.alias("tp"), rmf.alias("rmf")])
            .with_columns(
                # Calcula a diferença do TP
                pl.col("tp")
                .diff()
                .alias("tp_diff")
            )
            .with_columns(
                [
                    pl.when(pl.col("tp_diff") > 0)
                    .then(pl.col("rmf"))
                    .otherwise(0)
                    .alias("pmf"),
                    pl.when(pl.col("tp_diff") < 0)
                    .then(pl.col("rmf"))
                    .otherwise(0)
                    .alias("nmf"),
                ]
            )
        )

        # 4. Calcular a Soma do PMF e NMF no período
        data_with_sums = data_with_flows.with_columns(
            [
                pl.col("pmf")
                .rolling_sum(window_size=period, min_periods=period)
                .alias("pmf_sum"),
                pl.col("nmf")
                .rolling_sum(window_size=period, min_periods=period)
                .alias("nmf_sum"),
            ]
        )

        # 5. Calcular Money Flow Ratio (MFR)
        #    Evitar divisão por zero
        mfr = (
            pl.when(pl.col("nmf_sum") != 0)
            .then(pl.col("pmf_sum") / pl.col("nmf_sum"))
            .otherwise(None)
            .fill_null(pl.lit(float("inf")))
        )

        # 6. Calcular Money Flow Index (MFI)
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        # Correção para caso de mfr infinito (nmf_sum == 0), MFI deve ser 100
        mfi_final = pl.when(mfr == float("inf")).then(100.0).otherwise(mfi)

        # Montar resultado
        result_df = data_with_sums.select(
            [
                data.columns[0],  # Preserva a primeira coluna (índice/data)
                mfi_final.alias(f"MFI_{period}"),
            ]
        )

        return result_df
