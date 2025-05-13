import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class StochasticOscillatorIndicator(Indicator):
    """Calcula o Oscilador Estocástico (%K e %D).

    O Oscilador Estocástico compara o preço de fechamento de um ativo com
    seu range de preços durante um determinado período. Ele é usado para
    identificar níveis de sobrecompra e sobrevenda.
    %K = 100 * (Fechamento - Mínima(k)) / (Máxima(k) - Mínima(k))
    %D = SMA(%K, d)
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador Estocástico com sua configuração."""
        if config.type != IndicatorType.STOCH:
            raise ValueError(
                f"Configuração inválida para StochasticOscillatorIndicator. Tipo esperado: STOCH, recebido: {config.type}"
            )
        super().__init__(config)
        # Parâmetros: k_period, d_period
        self.k_period = int(self.config.params[0])
        self.d_period = int(self.config.params[1])
        # Colunas necessárias
        self.high_col = "high"
        self.low_col = "low"
        self.close_col = "close"

        # Nomes das colunas de saída
        self.param_str = f"{self.k_period}_{self.d_period}"
        self.k_col_name = f"StochK_{self.param_str}"
        self.d_col_name = f"StochD_{self.param_str}"

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula as linhas %K e %D do Oscilador Estocástico usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High', 'Low', 'Close'.
                  Deve estar ordenado por data.

        Returns:
            DataFrame Polars com as colunas %K e %D.
        """

        required = [self.high_col, self.low_col, self.close_col]
        if not all(col in data.columns for col in required):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required}"
            )

        # Calcula Mínima e Máxima do período k
        low_k = pl.col(self.low_col).rolling_min(
            window_size=self.k_period, min_periods=self.k_period
        )
        high_k = pl.col(self.high_col).rolling_max(
            window_size=self.k_period, min_periods=self.k_period
        )

        # Calcula %K
        # Evita divisão por zero se high_k == low_k
        k_series = data.with_columns(
            [low_k.alias("low_k"), high_k.alias("high_k")]
        ).select(
            pl.when(pl.col("high_k") == pl.col("low_k"))
            .then(0.0)  # Ou 50.0 ou None, dependendo da preferência
            .otherwise(
                100.0
                * (pl.col(self.close_col) - pl.col("low_k"))
                / (pl.col("high_k") - pl.col("low_k"))
            )
            .clip(0.0, 100.0)  # Garante que K esteja entre 0 e 100
            .alias(self.k_col_name)
        )

        # Calcula %D (SMA de %K)
        # Precisamos concatenar K para calcular D sobre ele
        temp_df_for_d = pl.concat(
            [data.select(data.columns[0]), k_series], how="horizontal"
        )
        d_series = temp_df_for_d.select(
            pl.col(self.k_col_name)
            .rolling_mean(window_size=self.d_period, min_periods=self.d_period)
            .alias(self.d_col_name)
        )

        # Combina K e D com a coluna de índice/tempo original
        if data.columns:
            result_df = pl.concat(
                [
                    data.select(data.columns[0]),  # Índice/Tempo
                    k_series,  # Coluna K
                    d_series,  # Coluna D
                ],
                how="horizontal",
            )
        else:  # Caso data não tenha colunas (improvável)
            result_df = pl.concat([k_series, d_series], how="horizontal")

        # Preencher NaNs iniciais (de rolling_min/max e rolling_mean)
        result_df = result_df.with_columns(
            [
                pl.all().forward_fill()  # Forward fill pode ser razoável aqui
                # .fill_null(strategy="backward") # Ou backward fill
            ]
        )

        return result_df
