import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class CCIIndicator(Indicator):
    """Calcula o Commodity Channel Index (CCI).

    O CCI mede a variação do preço de um ativo em relação à sua média estatística.
    Valores altos indicam que o preço está anormalmente alto em comparação com a média,
    e valores baixos indicam que está anormalmente baixo.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador CCI com sua configuração."""
        if config.type != IndicatorType.CCI:
            raise ValueError(
                f"Configuração inválida para CCIIndicator. Tipo esperado: CCI, recebido: {config.type}"
            )
        super().__init__(config)
        if not config.params or not isinstance(config.params[0], (int, float)):
            raise ValueError("O parâmetro 'period' (inteiro) é obrigatório para CCI.")
        self.period = int(config.params[0])
        # O fator 0.015 é padrão no cálculo do CCI
        self.constant = 0.015

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o Commodity Channel Index (CCI) usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High', 'Low', 'Close'. Deve estar ordenado por data.

        Returns:
            DataFrame Polars com a coluna 'CCI_{period}'.
        """
        required_cols = ["high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        period = self.period

        # 1. Calcular o Preço Típico (TP)
        data_with_tp = data.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("tp")
        )

        # 2. Calcular a Média Móvel Simples (SMA) do TP
        data_with_sma = data_with_tp.with_columns(
            pl.col("tp")
            .rolling_mean(window_size=period, min_periods=period)
            .alias("sma_tp")
        )

        # 3. Calcular o Desvio Médio (Mean Deviation)
        data_with_md = data_with_sma.with_columns(
            (pl.col("tp") - pl.col("sma_tp"))
            .abs()
            .rolling_mean(window_size=period, min_periods=period)
            .alias("mean_deviation")
        )

        # 4. Calcular o CCI
        #    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
        data_with_cci = data_with_md.with_columns(
            pl.when(pl.col("mean_deviation") != 0)
            .then(
                (pl.col("tp") - pl.col("sma_tp"))
                / (self.constant * pl.col("mean_deviation"))
            )
            # Ou 0.0 se preferir tratar divisão por zero como 0
            .otherwise(None)
            .alias(f"CCI_{period}")
        )

        # Selecionar e retornar a coluna CCI, mantendo a coluna de data/índice original
        if "date" not in data.columns:
            raise ValueError("Coluna 'date' não encontrada no DataFrame de entrada.")

        result_df = data_with_cci.select([
            pl.col("date"), # Selecionar explicitamente
            pl.col(f"CCI_{self.period}") # Usar self.period aqui
        ])

        return result_df
