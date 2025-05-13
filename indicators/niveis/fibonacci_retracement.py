import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------

FIB_LEVELS = {
    "0.0": 0.0,
    "23.6": 0.236,
    "38.2": 0.382,
    "50.0": 0.5,
    "61.8": 0.618,
    "100.0": 1.0,
}


class FibonacciRetracementIndicator(Indicator):
    """Calcula os níveis de Retração de Fibonacci.

    Identifica os pontos mais altos (High) e mais baixos (Low) em um determinado
    período e calcula os níveis de retração padrão (0%, 23.6%, 38.2%, 50%, 61.8%, 100%)
    entre esses dois pontos.
    """

    def __init__(self, config: IndicatorConfig):
        """Inicializa o indicador de Retração de Fibonacci com sua configuração."""
        if config.type != IndicatorType.FIBONACCI:
            raise ValueError(
                f"Configuração inválida para FibonacciRetracementIndicator. Tipo esperado: FIBONACCI, recebido: {config.type}"
            )
        super().__init__(config)
        if not config.params or not isinstance(config.params[0], (int)):
            raise ValueError(
                "O parâmetro 'period' (inteiro) é obrigatório para Fibonacci Retracement."
            )
        self.period = int(config.params[0])

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula os níveis de Retração de Fibonacci usando Polars.

        Args:
            data: DataFrame Polars com colunas 'High', 'Low'. Deve estar ordenado por data.

        Returns:
            DataFrame Polars com as colunas 'FIB_LEVEL_{level}_{period}' para cada nível
            de retração (0.0, 23.6, 38.2, 50.0, 61.8, 100.0).
        """
        required_cols = ["high", "low"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"DataFrame de entrada precisa conter as colunas: {required_cols}"
            )

        period = self.period

        # 1. Encontrar o High máximo e o Low mínimo na janela deslizante
        data_with_min_max = data.with_columns(
            [
                pl.col("high")
                .rolling_max(window_size=period, min_periods=period)
                .alias("highest_high"),
                pl.col("low")
                .rolling_min(window_size=period, min_periods=period)
                .alias("lowest_low"),
            ]
        )

        # 2. Calcular a diferença (range)
        data_with_range = data_with_min_max.with_columns(
            (pl.col("highest_high") - pl.col("lowest_low")).alias("price_range")
        )

        # 3. Calcular os níveis de Fibonacci
        fib_expressions = []
        for level_name, level_ratio in FIB_LEVELS.items():
            col_name = f"fib_{level_name}_{period}"
            # Nível = Low Mínimo + (Range * Ratio)
            # Usamos fill_null(0) no range para evitar erros se high == low, resultando no próprio low.
            expression = (
                pl.col("lowest_low")
                + (pl.col("price_range").fill_null(0) * level_ratio)
            ).alias(col_name)
            fib_expressions.append(expression)

        data_with_fib_levels = data_with_range.with_columns(fib_expressions)

        # Selecionar e retornar as colunas de Fibonacci, mantendo a coluna de data/índice original
        fib_col_names = [
            f"fib_{level_name}_{period}" for level_name in FIB_LEVELS.keys()
        ]
        result_df = data_with_fib_levels.select(
            [
                # Mantém a primeira coluna (geralmente data/índice)
                *(data.columns[:1] if data.columns else []),
                *fib_col_names,
            ]
        )

        return result_df
