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

        # Calcular a mudança no preço de fechamento
        close_diff = pl.col("close").diff()

        # Determinar o sinal do volume com base na mudança de preço
        # sign() retorna 1 para positivo, -1 para negativo, 0 para zero
        signed_volume = (
            pl.when(close_diff > 0)
            .then(pl.col("volume"))
            .when(close_diff < 0)
            .then(-pl.col("volume"))
            .otherwise(0)
        )  # Se não houve mudança no preço, OBV não muda

        # Calcular o OBV como a soma cumulativa do volume sinalizado
        # A primeira linha do OBV pode ser 0 ou o volume do primeiro dia.
        # Usaremos cumsum que começa com 0 implícito para a diferença.
        # Para ter o valor real, precisamos preencher o primeiro NaN.
        obv = signed_volume.cumsum(reverse=False).fill_null(
            0
        )  # Começa a acumulação do segundo dia
        # O primeiro dia tem OBV 0 por esta lógica.
        # Alternativa: iniciar com volume[0]

        # Adicionar coluna OBV ao DataFrame original
        result_df = data.with_columns(obv.alias("OBV")).select([data.columns[0], "OBV"])

        return result_df
