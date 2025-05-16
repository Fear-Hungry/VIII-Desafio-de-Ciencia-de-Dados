# data_loader/utils/adjustments.py
from typing import List, Optional

import polars as pl
import logging

# Configuração do logger
logger = logging.getLogger(__name__)

"""
Ajustes para eventos corporativos (splits e dividendos)
"""


def _ensure_datecol(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Garante que a coluna de data existe, é temporal e ordena o DataFrame."""
    if date_col not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(
            f"Coluna de data '{date_col}' não encontrada."
        )
    if df[date_col].dtype not in [pl.Date, pl.Datetime]:
        raise ValueError(
            f"Coluna de data '{date_col}' deve ser do tipo Date ou Datetime."
        )
    return df.sort(date_col)


def apply_splits(
    ohlcv_df: pl.DataFrame,
    splits_df: pl.DataFrame,
    date_col: str = "date",
    symbol_col: Optional[str] = "symbol",
    split_ratio_col: str = "split_ratio",
    price_cols: List[str] = ["open", "high", "low", "close"],
    volume_col: Optional[str] = "volume",
) -> pl.DataFrame:
    """
    Aplica ajustes de split a um DataFrame OHLCV de forma cumulativa, retroativa e vetorizada.

    Args:
        ohlcv_df (pl.DataFrame): DataFrame com dados OHLCV.
        splits_df (pl.DataFrame): DataFrame com splits.
        date_col (str): Nome da coluna de data.
        symbol_col (Optional[str]): Nome da coluna de símbolo, se múltiplos ativos.
        split_ratio_col (str): Nome da coluna com o fator de split (e.g., 2 para 2:1).
        price_cols (List[str]): Colunas de preço a ajustar.
        volume_col (Optional[str]): Coluna de volume a ajustar.

    Returns:
        pl.DataFrame: DataFrame ajustado retroativamente para splits.
    """
    if splits_df.is_empty() or split_ratio_col not in splits_df.columns:
        logger.warning(
            f"DataFrame de splits vazio ou coluna '{split_ratio_col}' não encontrada. Retornando DataFrame original."
        )
        return ohlcv_df

    # Garante colunas de data e ordena
    ohlcv_df = _ensure_datecol(ohlcv_df, date_col)
    splits_df = _ensure_datecol(splits_df, date_col)

    # Define as colunas de join
    join_on = (
        [symbol_col, date_col]
        if symbol_col
        and symbol_col in ohlcv_df.columns
        and symbol_col in splits_df.columns
        else [date_col]
    )
    if symbol_col and (
        symbol_col not in ohlcv_df.columns or symbol_col not in splits_df.columns
    ):
        logger.warning(
            f"Coluna de símbolo '{symbol_col}' não encontrada em ambos DataFrames. Tratando como ativo único."
        )
        join_on = [date_col]
        symbol_col = None  # Desativa o agrupamento por símbolo

    # 1. Junta os splits aos dados OHLCV
    df_joined = ohlcv_df.join(
        splits_df.select(join_on + [split_ratio_col]), on=join_on, how="left"
    ).with_columns(
        pl.col(split_ratio_col).fill_null(1.0)  # Fator 1 onde não há split
    )

    # Define a expressão de agrupamento (se houver símbolo)
    group_expr = [symbol_col] if symbol_col else []

    # 2. Calcula o fator cumulativo reverso
    # Ordena decrescente por data DENTRO de cada grupo (se houver)
    # Calcula cumprod e depois reordena crescente
    df_with_factor = (
        df_joined.sort(
            group_expr + [date_col], descending=[False] * len(group_expr) + [True]
        )
        .with_columns(
            pl.col(split_ratio_col)
            .cumprod()
            .over(group_expr)
            .alias("_split_factor_rev")
        )
        # O fator correto é o cumulativo dos splits futuros, então pegamos o valor do dia *seguinte* (shift(1) na ordem desc.)
        # e preenchemos o último valor (o mais recente no tempo, que é o primeiro na ordem desc.) com 1.0
        .with_columns(
            pl.col("_split_factor_rev")
            .shift(1)  # Alterado de -1 para 1
            .over(group_expr)
            .fill_null(1.0)
            .alias("_split_factor")
        )
        .sort(group_expr + [date_col])  # Reordena para a ordem original
    )

    # 3. Aplica o fator de ajuste
    expr_ajuste_preco = [
        (pl.col(col) / pl.col("_split_factor")).alias(col)
        for col in price_cols
        if col in df_with_factor.columns
    ]
    expr_ajuste_volume = []
    if volume_col and volume_col in df_with_factor.columns:
        expr_ajuste_volume = [
            (pl.col(volume_col) * pl.col("_split_factor")).alias(volume_col)
        ]

    # Verifica se alguma coluna de preço foi encontrada
    if not expr_ajuste_preco:
        logger.warning(
            f"Nenhuma das colunas de preço especificadas {price_cols} encontrada. Nenhum preço será ajustado."
        )

    ajustado = df_with_factor.with_columns(expr_ajuste_preco + expr_ajuste_volume).drop(
        [split_ratio_col, "_split_factor_rev", "_split_factor"]
    )  # Limpa colunas temporárias

    return ajustado


def apply_dividends(
    ohlcv_df: pl.DataFrame,
    dividends_df: pl.DataFrame,
    date_col: str = "date",
    symbol_col: Optional[str] = "symbol",
    dividend_amount_col: str = "dividend_amount",
    price_cols: List[str] = ["open", "high", "low", "close"],
    close_col_for_factor: str = "close",  # Coluna usada para calcular o fator
) -> pl.DataFrame:
    """
    Aplica ajustes de dividendos a um DataFrame OHLCV de forma cumulativa, retroativa e vetorizada.

    Args:
        ohlcv_df (pl.DataFrame): DataFrame com dados OHLCV.
        dividends_df (pl.DataFrame): DataFrame com dividendos.
        date_col (str): Nome da coluna de data (ex-dividend date no dividends_df).
        symbol_col (Optional[str]): Nome da coluna de símbolo, se múltiplos ativos.
        dividend_amount_col (str): Nome da coluna com o valor do dividendo.
        price_cols (List[str]): Colunas de preço a ajustar.
        close_col_for_factor (str): Coluna de preço do dia ANTERIOR à data ex-dividendo usada para calcular o fator. Geralmente 'close'.

    Returns:
        pl.DataFrame: DataFrame ajustado retroativamente para dividendos.
    """
    if dividends_df.is_empty() or dividend_amount_col not in dividends_df.columns:
        logger.warning(
            f"DataFrame de dividendos vazio ou coluna '{dividend_amount_col}' não encontrada. Retornando DataFrame original."
        )
        return ohlcv_df
    if close_col_for_factor not in ohlcv_df.columns:
        raise pl.exceptions.ColumnNotFoundError(
            f"Coluna '{close_col_for_factor}' necessária para cálculo do fator não encontrada no DataFrame OHLCV."
        )

    # Garante colunas de data e ordena
    ohlcv_df = _ensure_datecol(ohlcv_df, date_col)
    dividends_df = _ensure_datecol(dividends_df, date_col)

    # Define as colunas de join
    join_on = (
        [symbol_col, date_col]
        if symbol_col
        and symbol_col in ohlcv_df.columns
        and symbol_col in dividends_df.columns
        else [date_col]
    )
    if symbol_col and (
        symbol_col not in ohlcv_df.columns or symbol_col not in dividends_df.columns
    ):
        logger.warning(
            f"Coluna de símbolo '{symbol_col}' não encontrada em ambos DataFrames. Tratando como ativo único."
        )
        join_on = [date_col]
        symbol_col = None  # Desativa o agrupamento por símbolo

    # Define a expressão de agrupamento (se houver símbolo)
    group_expr = [symbol_col] if symbol_col else []

    # 1. Prepara dados OHLCV com o fechamento do dia anterior
    df_with_prev_close = ohlcv_df.with_columns(
        pl.col(close_col_for_factor).shift(1).over(group_expr).alias("_prev_close")
    )

    # 2. Junta os dividendos
    df_joined = df_with_prev_close.join(
        dividends_df.select(join_on + [dividend_amount_col]), on=join_on, how="left"
    )

    # 3. Calcula o fator de ajuste diário para dividendos
    # Fator = (prev_close - dividend) / prev_close, ou 1.0 se não houver dividendo ou prev_close for 0
    df_with_daily_factor = df_joined.with_columns(
        pl.when(
            (pl.col(dividend_amount_col).is_not_null()) & (pl.col("_prev_close") > 0)
        )
        .then(
            (pl.col("_prev_close") - pl.col(dividend_amount_col))
            / pl.col("_prev_close")
        )
        .otherwise(1.0)
        .alias("_daily_div_factor")
    )

    # 4. Calcula o fator cumulativo reverso
    df_with_cum_factor = (
        df_with_daily_factor.sort(
            group_expr + [date_col], descending=[False] * len(group_expr) + [True]
        )
        .with_columns(
            pl.col("_daily_div_factor")
            .cumprod()
            .over(group_expr)
            .alias("_div_factor_rev")
        )
        # O fator correto é o cumulativo dos ajustes futuros, então pegamos o valor do dia *seguinte* (shift(1) na ordem desc.)
        # e preenchemos o último valor (o mais recente no tempo, que é o primeiro na ordem desc.) com 1.0
        .with_columns(
            pl.col("_div_factor_rev")
            .shift(1)  # Alterado de -1 para 1
            .over(group_expr)
            .fill_null(1.0)
            .alias("_div_factor")
        )
        .sort(group_expr + [date_col])  # Reordena para a ordem original
    )

    # 5. Aplica o fator de ajuste aos preços
    expr_ajuste_preco = [
        (pl.col(col) * pl.col("_div_factor")).alias(col)
        for col in price_cols
        if col in df_with_cum_factor.columns
    ]

    # Verifica se alguma coluna de preço foi encontrada
    if not expr_ajuste_preco:
        logger.warning(
            f"Nenhuma das colunas de preço especificadas {price_cols} encontrada. Nenhum preço será ajustado."
        )

    ajustado = df_with_cum_factor.with_columns(expr_ajuste_preco).drop(
        [
            dividend_amount_col,
            "_prev_close",
            "_daily_div_factor",
            "_div_factor_rev",
            "_div_factor",
        ]
    )  # Limpa colunas temporárias

    return ajustado
