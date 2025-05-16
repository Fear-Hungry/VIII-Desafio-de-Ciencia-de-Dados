from typing import Union
import polars as pl

# --- Import relativo da classe base e config ---
from ..base import Indicator
from ..types import IndicatorConfig, IndicatorType

# -----------------------------------------------


class EMAIndicator(Indicator):
    """
    Média Móvel Exponencial (EMA).

    A EMA dá mais peso aos preços mais recentes, adaptando-se mais
    rapidamente a mudanças nos preços do que a SMA.

    Exemplos:
        >>> # Criar um indicador EMA de 12 períodos
        >>> ema = EMAIndicator(IndicatorConfig(type=IndicatorType.EMA, params=[12]))
        >>> # ou diretamente com o período
        >>> ema = EMAIndicator(12)
        >>> # Calcular o indicador para um DataFrame
        >>> df_com_ema = ema.calculate(df)
    """

    def __init__(self, config: Union[IndicatorConfig, int]):
        """
        Inicializa o indicador EMA.

        Args:
            config: Configuração do indicador (IndicatorConfig) ou diretamente
                   o período como inteiro.
        """
        if isinstance(config, int):
            # Se for só um número, assume que é o período
            self.span = config
            # Cria uma configuração padrão com o período fornecido
            self.config = IndicatorConfig(
                type=IndicatorType.EMA,
                params=[self.span]
            )
        else:
            # Assume que é um IndicatorConfig
            self.config = config
            self.span = self.config.params[0] if self.config.params else 12

        # Nome para a coluna de saída
        self.output_column = f"ema_{self.span}"

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula o indicador EMA para o DataFrame fornecido.

        Args:
            df: DataFrame com dados OHLCV.

        Returns:
            DataFrame com a coluna do indicador EMA adicionada.
        """
        if "close" not in df.columns:
            raise ValueError("A coluna 'close' é necessária para calcular o EMA")
        if "date" not in df.columns:
            raise ValueError("A coluna 'date' é necessária para o resultado do EMA")

        # Calcula EMA e seleciona a coluna de data junto com a nova coluna EMA
        # O .ewm_mean() já retorna uma Series quando usado em um select/with_columns.
        # O parâmetro `span` é mais comum para EMA do que `alpha` diretamente.
        # `adjust=False` é o comportamento padrão para EMAs em finanças (usa os pesos corretos desde o início).
        result_df = df.select([
            pl.col("date"),
            pl.col("close").ewm_mean(span=self.span, adjust=False).alias(self.output_column)
        ])

        return result_df

    def __str__(self) -> str:
        return f"EMA({self.span})"
