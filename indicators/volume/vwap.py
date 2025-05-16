import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class VWAPIndicator(Indicator):
    """Calcula o Volume Weighted Average Price (VWAP) de forma rolante.
    Nota: O VWAP padrão é geralmente calculado intra-diário e reiniciado a cada dia.
    Esta implementação calcula um VWAP rolante sobre um período definido.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador VWAP rolante."""
        if config.type != IndicatorType.VWAP:  # Assume IndicatorType.VWAP
            raise ValueError(
                f"Configuração inválida para VWAPIndicator. Tipo esperado: VWAP, recebido: {config.type}"
            )
        super().__init__(config)
        if not config.params or not isinstance(config.params[0], (int, float)):
            raise ValueError("Parâmetro 'period' inválido ou ausente para VWAP.")
        self.period = int(config.params[0])

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o VWAP rolante usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High', 'Low', 'Close', 'Volume'.

        Returns:
            DataFrame Polars com a coluna 'VWAP_{period}'.
        """
        required_cols = ["high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        period = self.period

        # 1. Calcular Typical Price (TP)
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3

        # 2. Calcular TP * Volume
        tp_vol = tp * pl.col("volume")

        # 3. Calcular somas rolantes de (TP * Volume) e Volume
        sum_tp_vol = tp_vol.rolling_sum(window_size=period, min_periods=period)
        sum_vol = pl.col("volume").rolling_sum(window_size=period, min_periods=period)

        # 4. Calcular VWAP rolante
        #    Evitar divisão por zero
        vwap = (
            pl.when(sum_vol != 0)
            .then(sum_tp_vol / sum_vol)
            .otherwise(None)
            .fill_null(strategy="forward")
        )

        # Montar resultado
        if "date" not in data.columns:
            raise ValueError("Coluna 'date' não encontrada no DataFrame de entrada.")

        # O nome da coluna do VWAP deve incluir o período para evitar conflitos
        # se diferentes períodos de VWAP forem calculados.
        vwap_col_name = f"VWAP_{self.period}"

        result_df = data.with_columns(vwap.alias(vwap_col_name)).select(
            [pl.col("date"), pl.col(vwap_col_name)]
        )

        return result_df
