import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class IchimokuCloudIndicator(Indicator):
    """Calcula as linhas da Nuvem Ichimoku.

    Inclui Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B e Chikou Span.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador Ichimoku Cloud com sua configuração."""
        if config.type != IndicatorType.ICHIMOKU:
            raise ValueError(
                f"Configuração inválida para IchimokuCloudIndicator. Tipo esperado: ICHIMOKU, recebido: {config.type}"
            )
        super().__init__(config)

        if not config.params or len(config.params) != 3:
            raise ValueError(
                "Ichimoku Cloud requer 3 parâmetros: [tenkan_period, kijun_period, senkou_span_b_period]."
            )

        try:
            self.tenkan_period = int(config.params[0])
            self.kijun_period = int(config.params[1])
            self.senkou_span_b_period = int(config.params[2])
        except (ValueError, TypeError):
            raise ValueError("Os parâmetros para Ichimoku Cloud devem ser inteiros.")

        # O deslocamento para Senkou Spans e Chikou Span é geralmente o período Kijun
        self.displacement = self.kijun_period

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula as linhas da Nuvem Ichimoku usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High', 'Low', 'Close'. Deve estar ordenado por data.

        Returns:
            DataFrame Polars com as colunas Ichimoku:
            'TENKAN_{p1}', 'KIJUN_{p2}', 'SENKOU_A_{p1}_{p2}',
            'SENKOU_B_{p3}', 'CHIKOU_{p2}'.
        """
        required_cols = ["high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        p1 = self.tenkan_period
        p2 = self.kijun_period
        p3 = self.senkou_span_b_period
        displacement = self.displacement

        # 1. Calcular Tenkan-sen (Conversion Line)
        high_p1 = pl.col("high").rolling_max(window_size=p1, min_periods=p1)
        low_p1 = pl.col("low").rolling_min(window_size=p1, min_periods=p1)
        tenkan_sen = ((high_p1 + low_p1) / 2).alias(f"TENKAN_{p1}")

        # 2. Calcular Kijun-sen (Base Line)
        high_p2 = pl.col("high").rolling_max(window_size=p2, min_periods=p2)
        low_p2 = pl.col("low").rolling_min(window_size=p2, min_periods=p2)
        kijun_sen = ((high_p2 + low_p2) / 2).alias(f"KIJUN_{p2}")

        # Calcular componentes antes de aplicar deslocamentos
        data_with_lines = data.with_columns([tenkan_sen, kijun_sen])

        # 3. Calcular Senkou Span A (Leading Span A)
        #    É a média de Tenkan e Kijun, DESLOCADA PARA FRENTE
        senkou_a_calc = ((pl.col(f"TENKAN_{p1}") + pl.col(f"KIJUN_{p2}")) / 2).shift(
            displacement
        )
        senkou_a = senkou_a_calc.alias(f"SENKOU_A_{p1}_{p2}")

        # 4. Calcular Senkou Span B (Leading Span B)
        #    É a média do High/Low do período p3, DESLOCADA PARA FRENTE
        high_p3 = pl.col("high").rolling_max(window_size=p3, min_periods=p3)
        low_p3 = pl.col("low").rolling_min(window_size=p3, min_periods=p3)
        senkou_b_calc = ((high_p3 + low_p3) / 2).shift(displacement)
        senkou_b = senkou_b_calc.alias(f"SENKOU_B_{p3}")

        # 5. Calcular Chikou Span (Lagging Span)
        #    É o preço de fechamento, DESLOCADO PARA TRÁS
        chikou_span = pl.col("close").shift(-displacement).alias(f"CHIKOU_{p2}")

        # Adicionar linhas deslocadas ao DataFrame
        data_with_all_lines = data_with_lines.with_columns(
            [senkou_a, senkou_b, chikou_span]
        )

        # Selecionar e retornar as colunas Ichimoku, mantendo a coluna de data/índice original
        ichimoku_cols = [
            f"TENKAN_{p1}",
            f"KIJUN_{p2}",
            f"SENKOU_A_{p1}_{p2}",
            f"SENKOU_B_{p3}",
            f"CHIKOU_{p2}",
        ]

        result_df = data_with_all_lines.select(
            [
                pl.col("date"),
                *ichimoku_cols,
            ]
        )

        return result_df
