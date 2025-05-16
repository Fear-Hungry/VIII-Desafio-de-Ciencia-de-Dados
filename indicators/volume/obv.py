import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class OBVIndicator(Indicator):
    """Calcula o On-Balance Volume (OBV)."""

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador OBV."""
        if config.type != IndicatorType.OBV:  # Assume IndicatorType.OBV
            raise ValueError(
                f"Configuração inválida para OBVIndicator. Tipo esperado: OBV, recebido: {config.type}"
            )
        # OBV geralmente não tem parâmetros como período
        super().__init__(config)
        # Poderíamos verificar se config.params está vazio, mas vamos permitir por flexibilidade

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o OBV usando Polars.

        Args:
            data: DataFrame Polars com colunas 'Close', 'Volume'.

        Returns:
            DataFrame Polars com a coluna 'OBV'.
        """
        required_cols = ["close", "volume"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        # Calcula a mudança no preço de fechamento e o volume sinalizado
        # como novas colunas temporárias
        temp_df = data.with_columns(
            pl.col("close").diff().alias("_close_diff_temp_")
        ).with_columns(
            pl.when(pl.col("_close_diff_temp_") > 0).then(pl.col("volume"))
            .when(pl.col("_close_diff_temp_") < 0).then(-pl.col("volume"))
            .otherwise(0).alias("_signed_volume_temp_")
        )

        # Calcula o OBV usando cumsum na nova coluna e seleciona
        # A primeira coluna de 'data' (data.columns[0]) é geralmente a coluna de data/tempo
        result_df = temp_df.with_columns(
            pl.col("_signed_volume_temp_").cumsum().fill_null(0).alias("OBV")
        ).select([data.columns[0], "OBV"])

        return result_df
