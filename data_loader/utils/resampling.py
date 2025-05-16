import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import polars as pl
from polars.datatypes import Date, Datetime
import logging

# Adiciona verificação de tipo para evitar erro de importação circular
# ou dependência desnecessária se só usado para type hinting.
if TYPE_CHECKING:
    from polars.type_aliases import PolarsExpr

# Configuração do logger
logger = logging.getLogger(__name__)


"""
Funções auxiliares para reamostragem de dados
"""

# Constantes para períodos comuns de reamostragem


class TimeFrames(str, Enum):
    """Períodos de tempo comuns para reamostragem."""

    MINUTO_1 = "1m"
    MINUTO_5 = "5m"
    MINUTO_15 = "15m"
    MINUTO_30 = "30m"
    HORA_1 = "1h"
    HORA_4 = "4h"
    DIARIO = "1D"
    SEMANAL = "1w"
    MENSAL = "1mo"
    TRIMESTRAL = "3mo"
    ANUAL = "1y"


# Mapeamento para calendários específicos - REMOVIDO PARA MODULARIDADE
# CALENDARIOS = {
#     "B3": "Brazil/BOVESPA",
#     "NYSE": "US/NYSE",
#     "NASDAQ": "US/NASDAQ",
#     "LSE": "UK/LSE"
# }


def _validate_time_column(data: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Valida se a coluna de tempo existe, é do tipo temporal e ordena os dados por ela."""
    if time_col not in data.columns:
        raise pl.exceptions.ColumnNotFoundError(
            f"Coluna de tempo '{time_col}' não encontrada no DataFrame."
        )
    if data[time_col].dtype not in [pl.Date, pl.Datetime]:
        raise ValueError(
            f"Coluna de tempo '{time_col}' deve ser do tipo Date ou Datetime."
        )
    return data.sort(time_col)


def resample_ohlc(
    data: pl.DataFrame,
    time_col: str,
    timeframe: str = "1D",
    price_cols: List[str] = ["open", "high", "low", "close"],
    volume_cols: List[str] = ["volume"],
    additional_cols: Optional[Dict[str, "PolarsExpr"]] = None,
) -> pl.DataFrame:
    """
    Reamostra um DataFrame OHLC(V) para um timeframe diferente usando Polars.
    Os dados de entrada SÃO ordenados pela time_col dentro desta função.

    Args:
        data (pl.DataFrame): DataFrame Polars contendo os dados OHLC(V).
        time_col (str): Nome da coluna que contém os timestamps (deve ser tipo temporal).
        timeframe (str): O intervalo de tempo para reamostragem (ex: '1H', '1D', '1W').
                         Usa a sintaxe de offset do Polars/Pandas.
        price_cols (List[str]): Lista dos nomes das colunas de preço, na ordem:
                                [open, high, low, close].
        volume_cols (List[str]): Lista dos nomes das colunas de volume.
        additional_cols (Optional[Dict[str, PolarsExpr]]): Dicionário opcional para agregar
                                                        colunas adicionais. As chaves são os
                                                        nomes das colunas de origem e os valores
                                                        são as expressões de agregação do Polars
                                                        (ex: {'minha_col': pl.sum('minha_col')}).
                                                        A expressão deve incluir o alias desejado
                                                        para a coluna resultante (ex: pl.sum('col').alias('col_sum')).

    Returns:
        pl.DataFrame: DataFrame Polars reamostrado.

    Raises:
        ValueError: Se `price_cols` não contiver pelo menos 4 colunas ou
                    se a coluna de tempo não for encontrada ou não for temporal.
        pl.exceptions.ColumnNotFoundError: Se alguma coluna especificada não existir.
    """
    data = _validate_time_column(data, time_col)

    if len(price_cols) < 4:
        raise ValueError(
            "`price_cols` deve conter pelo menos 4 colunas: open, high, low, close."
        )

    # Garante que a coluna de tempo seja a primeira para group_by_dynamic (convenção)
    # Embora não estritamente necessário se `index_column` for usado.
    # data = data.select([time_col] + [col for col in data.columns if col != time_col])

    # Cria a lista de expressões de agregação
    agg_exprs = []

    # Agregações padrão para OHLC
    if price_cols[0] in data.columns:
        agg_exprs.append(pl.first(price_cols[0]).alias("open"))
    else:
        raise pl.exceptions.ColumnNotFoundError(
            f"Coluna open '{price_cols[0]}' não encontrada."
        )

    if price_cols[1] in data.columns:
        agg_exprs.append(pl.max(price_cols[1]).alias("high"))
    else:
        raise pl.exceptions.ColumnNotFoundError(
            f"Coluna high '{price_cols[1]}' não encontrada."
        )

    if price_cols[2] in data.columns:
        agg_exprs.append(pl.min(price_cols[2]).alias("low"))
    else:
        raise pl.exceptions.ColumnNotFoundError(
            f"Coluna low '{price_cols[2]}' não encontrada."
        )

    if price_cols[3] in data.columns:
        agg_exprs.append(pl.last(price_cols[3]).alias("close"))
    else:
        raise pl.exceptions.ColumnNotFoundError(
            f"Coluna close '{price_cols[3]}' não encontrada."
        )

    # Agregação para colunas de volume
    for col in volume_cols:
        if col in data.columns:
            agg_exprs.append(pl.sum(col).alias(col))
        else:
            logger.warning(
                f"Coluna de volume '{col}' não encontrada. Pulando agregação."
            )

    # Agregações para colunas adicionais usando expressões Polars
    if additional_cols:
        for col_origin, agg_expr in additional_cols.items():
            if col_origin in data.columns:
                # Adiciona a expressão diretamente
                agg_exprs.append(agg_expr)
            else:
                logger.warning(
                    f"Coluna de origem '{col_origin}' para agregação adicional não encontrada. Pulando."
                )

    # Realiza a reamostragem usando group_by_dynamic
    resampled_df = data.group_by_dynamic(
        index_column=time_col,
        every=timeframe,
        period=timeframe,  # Garante que os períodos cubram o 'every'
        offset=datetime.timedelta(0),  # Offset inicial, padrão 0
        closed="left",  # Intervalo fechado à esquerda [start, end)
    ).agg(
        # Passa as expressões de agregação desempacotadas
        *agg_exprs
    )

    # Ordena o resultado final pelo tempo (boa prática)
    resampled_df = resampled_df.sort(time_col)

    return resampled_df


def get_timeframe(
    freq: str,
    start_date: datetime.datetime | datetime.date,
    end_date: datetime.datetime | datetime.date,
) -> pl.Series:
    """
    Gera uma série temporal (Polars Series) dentro de um intervalo de datas.

    Args:
        freq (str): A frequência do intervalo de tempo (ex: '1H', '1D', '1mo').
                    Usa a sintaxe de offset do Polars/Pandas.
        start_date (datetime.datetime | datetime.date): Data/hora de início.
        end_date (datetime.datetime | datetime.date): Data/hora de fim (inclusiva).

    Returns:
        pl.Series: Uma série Polars contendo os timestamps gerados.
    """
    # Eager=True retorna uma Series diretamente
    return pl.date_range(start=start_date, end=end_date, interval=freq, eager=True)


def align_multiple_timeframes(
    data: pl.DataFrame,
    time_col: str,
    main_timeframe: str = "1D",
    resampled_timeframes: Optional[Dict[str, str]] = None,
    price_cols: List[str] = ["open", "high", "low", "close"],
    volume_cols: List[str] = ["volume"],
    additional_cols: Optional[Dict[str, "PolarsExpr"]] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Reamostra um DataFrame para múltiplos timeframes.

    Args:
        data (pl.DataFrame): O DataFrame original com dados de maior frequência.
        time_col (str): Nome da coluna de tempo no DataFrame original.
        main_timeframe (str): O timeframe principal que sempre será incluído
                               no resultado (mesmo que não esteja em
                               `resampled_timeframes`). (Padrão: '1D').
        resampled_timeframes (Optional[Dict[str, str]]): Um dicionário onde as chaves
                                                        são nomes para os timeframes
                                                        (ex: 'Diário', 'Semanal') e os
                                                        valores são as strings de
                                                        timeframe do Polars (ex: '1D', '1W').
                                                        Se None, apenas o `main_timeframe`
                                                        será retornado.
        price_cols (List[str]): Passado para `resample_ohlc`.
        volume_cols (List[str]): Passado para `resample_ohlc`.
        additional_cols (Optional[Dict[str, str]]): Passado para `resample_ohlc`.


    Returns:
        Dict[str, pl.DataFrame]: Um dicionário onde as chaves são os nomes dos
                                 timeframes (incluindo o `main_timeframe` com
                                 sua própria string como chave se não estiver em
                                 `resampled_timeframes`) e os valores são os
                                 DataFrames Polars reamostrados correspondentes.
    """
    _validate_time_column(data, time_col)

    result = {}

    # Garante que o timeframe principal esteja presente
    all_timeframes = {main_timeframe: main_timeframe}  # {nome: freq}
    if resampled_timeframes:
        all_timeframes.update(resampled_timeframes)

    for name, timeframe in all_timeframes.items():
        try:
            # Passa additional_cols como expressões se fornecido
            current_additional_cols = additional_cols if additional_cols else None
            result[name] = resample_ohlc(
                data=data,
                time_col=time_col,
                timeframe=timeframe,
                price_cols=price_cols,
                volume_cols=volume_cols,
                additional_cols=current_additional_cols,
            )
        except Exception as e:
            logger.error(
                f"Erro ao reamostrar para o timeframe '{name}' ({timeframe}): {e}",
                exc_info=True,
            )
            # Decide se continua ou relança o erro
            # result[name] = None # Ou pode pular a chave

    return result


def get_business_days(
    start_date: datetime.date,
    end_date: datetime.date,
    holidays: Optional[List[datetime.date]] = None,
) -> pl.Series:
    """
    Gera uma série de dias úteis (segunda a sexta, opcionalmente excluindo feriados)
    dentro de um intervalo.

    Args:
        start_date (datetime.date): Data de início.
        end_date (datetime.date): Data de fim (inclusiva).
        holidays (Optional[List[datetime.date]]): Uma lista opcional de datas
                                                  (feriados) a serem excluídas
                                                  dos dias úteis. Default é None (não exclui feriados).

    Returns:
        pl.Series: Uma série Polars contendo os dias úteis.

    Nota:
        Esta função considera dias úteis como segunda a sexta-feira.
        Para uma gestão de calendários de mercado mais robusta (incluindo feriados específicos
        de bolsas), considere usar bibliotecas especializadas como pandas_market_calendars
        e passar a lista de feriados obtida para o parâmetro `holidays`.
    """
    # Cria uma série de dias
    days = pl.date_range(
        start=start_date, end=end_date, interval="1d", eager=True, name="date"
    )  # Nomear a coluna é importante

    # Filtra apenas os dias úteis (segunda a sexta)
    business_days = days.filter(pl.col("date").dt.weekday() < 5)

    # Se uma lista de feriados foi fornecida, filtra esses dias
    if holidays:
        # Garante que a lista de feriados não esteja vazia para evitar erros no `is_in`
        if holidays:
            business_days = business_days.filter(~pl.col("date").is_in(holidays))
        else:
            # Se a lista está vazia, não faz nada
            pass

    return business_days


def resample_to_common_periods(
    data: pl.DataFrame,
    time_col: str,
    price_cols: List[str] = ["open", "high", "low", "close"],
    volume_cols: List[str] = ["volume"],
    include_intraday: bool = False,
    additional_cols: Optional[Dict[str, "PolarsExpr"]] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Reamostra os dados para os períodos mais comuns: diário, semanal e mensal.
    Opcionalmente também incluí períodos intradiários.

    Args:
        data (pl.DataFrame): DataFrame Polars contendo os dados OHLC(V).
        time_col (str): Nome da coluna que contém os timestamps.
        price_cols (List[str]): Lista dos nomes das colunas de preço, na ordem OHLC.
        volume_cols (List[str]): Lista dos nomes das colunas de volume.
        include_intraday (bool): Se True, inclui timeframes intradiários (1h, 4h).

    Returns:
        Dict[str, pl.DataFrame]: Um dicionário com os DataFrames reamostrados para cada período.
    """
    _validate_time_column(data, time_col)

    # Define os timeframes a serem incluídos
    timeframes = {
        "Diário": TimeFrames.DIARIO,
        "Semanal": TimeFrames.SEMANAL,
        "Mensal": TimeFrames.MENSAL,
    }

    # Adiciona timeframes intradiários se solicitado
    if include_intraday:
        timeframes.update({"1_Hora": TimeFrames.HORA_1, "4_Horas": TimeFrames.HORA_4})

    # Mapeia nomes de volta para strings de timeframe do Polars para a chamada
    tf_strings_dict = {name: tf_enum for name, tf_enum in timeframes.items()}

    return align_multiple_timeframes(
        data=data,
        time_col=time_col,
        main_timeframe=TimeFrames.DIARIO,
        resampled_timeframes=tf_strings_dict,
        price_cols=price_cols,
        volume_cols=volume_cols,
        additional_cols=additional_cols,
    )


def convert_timeframe_notation(timeframe: str, to_format: str = "polars") -> str:
    """
    Converte entre diferentes notações de timeframe.

    Args:
        timeframe (str): A string do timeframe a ser convertida.
        to_format (str): O formato de destino. Opções: "polars", "pandas", "tradingview".

    Returns:
        str: O timeframe convertido para o formato especificado.

    Exemplos:
        >>> convert_timeframe_notation("1D", "tradingview")
        "1day"
        >>> convert_timeframe_notation("4h", "pandas")
        "4H"
    """
    # Mapeamento entre formatos
    mappings = {
        # Polars para outros formatos
        "polars_to_tradingview": {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "4h": "240",
            "1D": "1day",
            "1d": "1day",
            "1w": "1week",
            "1W": "1week",
            "1mo": "1month",
            "1y": "12month",
        },
        "polars_to_pandas": {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1H",
            "4h": "4H",
            "1D": "1D",
            "1d": "1D",
            "1w": "1W",
            "1W": "1W",
            "1mo": "1M",
            "1y": "1Y",
        },
        # TradingView para outros formatos
        "tradingview_to_polars": {
            "1": "1m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "240": "4h",
            "1day": "1D",
            "1week": "1w",
            "1month": "1mo",
            "12month": "1y",
        },
        # Pandas para outros formatos
        "pandas_to_polars": {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1H": "1h",
            "4H": "4h",
            "1D": "1D",
            "1W": "1w",
            "1M": "1mo",
            "1Y": "1y",
        },
    }

    # Detectar formato de entrada
    input_format = "polars"  # formato padrão
    for fmt in ["polars", "tradingview", "pandas"]:
        for k in mappings.get(f"{fmt}_to_polars", {}).keys():
            if timeframe.lower() == k.lower():
                input_format = fmt
                break
        if input_format != "polars":
            break

    # Se o formato de entrada e saída forem iguais, retorna o mesmo valor
    if input_format == to_format:
        return timeframe

    # Converte para polars primeiro (formato intermediário)
    if input_format != "polars":
        mapping_key = f"{input_format}_to_polars"
        if mapping_key in mappings:
            # Tenta correspondência direta
            if timeframe in mappings[mapping_key]:
                polars_timeframe = mappings[mapping_key][timeframe]
            else:
                # Tenta correspondência case-insensitive
                for k, v in mappings[mapping_key].items():
                    if timeframe.lower() == k.lower():
                        polars_timeframe = v
                        break
                else:
                    # Se não encontrar, retorna o original
                    return timeframe
        else:
            return timeframe
    else:
        polars_timeframe = timeframe

    # Converte de polars para o formato de destino
    if to_format != "polars":
        mapping_key = f"polars_to_{to_format}"
        if mapping_key in mappings and polars_timeframe in mappings[mapping_key]:
            return mappings[mapping_key][polars_timeframe]

    # Se chegou aqui, retorna o formato polars
    return polars_timeframe


def create_compound_timeframe_dataset(
    data: pl.DataFrame,
    time_col: str,
    price_cols: List[str] = ["open", "high", "low", "close"],
    volume_cols: List[str] = ["volume"],
    timeframes: List[str] = [
        TimeFrames.DIARIO.value,
        TimeFrames.SEMANAL.value,
        TimeFrames.MENSAL.value,
    ],
    suffix_format: str = "{}_{}",
    join_type: str = "inner",
    additional_cols: Optional[Dict[str, "PolarsExpr"]] = None,
) -> pl.DataFrame:
    """
    Cria um DataFrame composto com dados de múltiplos timeframes unidos em um único DataFrame.

    Args:
        data (pl.DataFrame): DataFrame Polars contendo os dados OHLC(V).
        time_col (str): Nome da coluna que contém os timestamps.
        price_cols (List[str]): Lista dos nomes das colunas de preço, na ordem OHLC.
        volume_cols (List[str]): Lista dos nomes das colunas de volume.
        timeframes (List[str]): Lista de timeframes a serem incluídos.
        suffix_format (str): Formato para o sufixo das colunas. Deve conter dois
                            placeholders {} para o nome da coluna e timeframe.
        join_type (str): Tipo de join a ser usado. Opções: "inner", "left", "outer".

    Returns:
        pl.DataFrame: Um DataFrame único com os dados de todos os timeframes unidos.
    """
    _validate_time_column(data, time_col)

    resampled_dfs = {}
    for tf in timeframes:
        resampled_dfs[tf] = resample_ohlc(
            data=data,
            time_col=time_col,
            timeframe=tf,
            price_cols=price_cols,
            volume_cols=volume_cols,
            additional_cols=additional_cols,
        )

        # Renomeia as colunas para evitar conflitos (exceto time_col)
        for col in resampled_dfs[tf].columns:
            if col != time_col:
                resampled_dfs[tf] = resampled_dfs[tf].rename(
                    {col: suffix_format.format(col, tf.replace("1", ""))}
                )

    # Inicializa o DataFrame resultante com o primeiro timeframe
    result_df = resampled_dfs[timeframes[0]]

    # Une os demais timeframes
    for tf in timeframes[1:]:
        if join_type == "inner":
            result_df = result_df.join(resampled_dfs[tf], on=time_col, how="inner")
        elif join_type == "left":
            result_df = result_df.join(resampled_dfs[tf], on=time_col, how="left")
        elif join_type == "outer":
            result_df = result_df.join(resampled_dfs[tf], on=time_col, how="outer")

    return result_df.sort(time_col)


def split_timeframe_by_session(
    data: pl.DataFrame,
    time_col: str,
    session_start: str = "10:00:00",
    session_end: str = "17:00:00",
    price_cols: List[str] = ["open", "high", "low", "close"],
    volume_cols: List[str] = ["volume"],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Divide um DataFrame em dois conjuntos: sessão regular e fora da sessão.

    Args:
        data (pl.DataFrame): DataFrame Polars contendo os dados OHLC(V).
        time_col (str): Nome da coluna que contém os timestamps.
        session_start (str): Hora de início da sessão regular (formato "HH:MM:SS").
        session_end (str): Hora de fim da sessão regular (formato "HH:MM:SS").
        price_cols (List[str]): Lista dos nomes das colunas de preço.
        volume_cols (List[str]): Lista dos nomes das colunas de volume.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Uma tupla com (dados_sessao_regular, dados_fora_sessao).
    """
    data = _validate_time_column(data, time_col)

    # Converte as strings de hora para objetos time
    start_time = datetime.time.fromisoformat(session_start)
    end_time = datetime.time.fromisoformat(session_end)

    # Filtra os dados da sessão regular
    regular_session = data.filter(
        (pl.col(time_col).dt.time() >= start_time)
        & (pl.col(time_col).dt.time() <= end_time)
    )

    # Filtra os dados fora da sessão regular
    non_regular_session = data.filter(
        (pl.col(time_col).dt.time() < start_time)
        | (pl.col(time_col).dt.time() > end_time)
    )

    return regular_session, non_regular_session


def align_multiple_assets(
    asset_data: Dict[str, pl.DataFrame], time_col: str, join_type: str = "outer"
) -> pl.DataFrame:
    """
    Alinha temporalmente DataFrames de múltiplos ativos em um único DataFrame.

    Args:
        asset_data (Dict[str, pl.DataFrame]): Dicionário onde as chaves são os nomes/símbolos
                                            dos ativos e os valores são os DataFrames Polars
                                            correspondentes. Espera-se que cada DataFrame
                                            contenha a coluna de tempo especificada.
        time_col (str): Nome da coluna de tempo usada para o alinhamento.
        join_type (str): Tipo de join a ser usado ao combinar os DataFrames.
                         Opções comuns: "outer", "inner", "left". Default: "outer".

    Returns:
        pl.DataFrame: Um DataFrame Polars único contendo os dados alinhados de todos os ativos.
                      As colunas originais (exceto `time_col`) serão renomeadas com o nome
                      do ativo como prefixo (ex: "ATIVO_close", "ATIVO_volume").

    Raises:
        ValueError: Se o dicionário `asset_data` estiver vazio.
    """
    if not asset_data:
        raise ValueError("O dicionário 'asset_data' não pode estar vazio.")

    # Pega o primeiro ativo como base inicial
    first_asset_name = next(iter(asset_data))
    aligned_df = asset_data[first_asset_name]

    # Renomeia as colunas do primeiro ativo (exceto time_col)
    rename_mapping = {
        col: f"{first_asset_name}_{col}"
        for col in aligned_df.columns
        if col != time_col
    }
    aligned_df = aligned_df.rename(rename_mapping)

    # Itera sobre os ativos restantes e faz o join
    for asset_name, df in list(asset_data.items())[1:]:
        # Renomeia as colunas do DataFrame atual antes do join
        rename_mapping = {
            col: f"{asset_name}_{col}" for col in df.columns if col != time_col
        }
        df_renamed = df.rename(rename_mapping)

        # Realiza o join com o DataFrame alinhado
        aligned_df = aligned_df.join(df_renamed, on=time_col, how=join_type)

    # Ordena pelo tempo por garantia
    aligned_df = aligned_df.sort(time_col)

    return aligned_df


def handle_outliers(
    data: pl.DataFrame,
    columns: List[str],
    method: str = "iqr",
    treatment: str = "cap",
    threshold: float = 1.5,
) -> pl.DataFrame:
    """
    Detecta e trata outliers em colunas especificadas de um DataFrame Polars.

    Args:
        data (pl.DataFrame): O DataFrame de entrada.
        columns (List[str]): Lista de nomes das colunas onde os outliers serão tratados.
        method (str): Método de detecção de outliers. Atualmente suportado: "iqr".
                      Default: "iqr".
        treatment (str): Método de tratamento dos outliers detectados. Opções:
                         "cap": Substitui outliers pelos limites (Q1 - th*IQR, Q3 + th*IQR).
                         "null": Substitui outliers por None/Null.
                         Default: "cap".
        threshold (float): O multiplicador para o método IQR. Default: 1.5.

    Returns:
        pl.DataFrame: DataFrame com os outliers tratados nas colunas especificadas.

    Raises:
        ValueError: Se um método ou tratamento não suportado for especificado.
        pl.exceptions.ColumnNotFoundError: Se alguma coluna especificada não existir.
    """
    if method.lower() != "iqr":
        raise ValueError(f"Método de detecção '{method}' não suportado. Use 'iqr'.")
    if treatment.lower() not in ["cap", "null"]:
        raise ValueError(
            f"Método de tratamento '{treatment}' não suportado. Use 'cap' ou 'null'."
        )

    df_processed = data.clone()  # Evita modificar o DataFrame original

    for col_name in columns:
        if col_name not in df_processed.columns:
            raise pl.exceptions.ColumnNotFoundError(
                f"Coluna '{col_name}' não encontrada."
            )

        # Calcula Q1, Q3 e IQR apenas para valores não nulos
        q1 = df_processed.select(pl.col(col_name).quantile(0.25, "linear")).item()
        q3 = df_processed.select(pl.col(col_name).quantile(0.75, "linear")).item()

        # Verifica se q1 ou q3 são None (pode ocorrer se a coluna tiver poucos dados ou só nulos)
        if q1 is None or q3 is None:
            logger.warning(
                f"Não foi possível calcular Q1/Q3 para a coluna '{col_name}'. Pulando tratamento de outliers."
            )
            continue

        iqr = q3 - q1

        # Define os limites inferior e superior
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        # Cria a expressão de condição para outliers
        is_outlier = (pl.col(col_name) < lower_bound) | (pl.col(col_name) > upper_bound)

        if treatment.lower() == "cap":
            # Aplica o capping (limitar aos bounds)
            df_processed = df_processed.with_columns(
                pl.when(is_outlier & (pl.col(col_name) < lower_bound))
                # Substitui outliers inferiores pelo limite inferior
                .then(pl.lit(lower_bound))
                .when(is_outlier & (pl.col(col_name) > upper_bound))
                # Substitui outliers superiores pelo limite superior
                .then(pl.lit(upper_bound))
                .otherwise(pl.col(col_name))  # Mantém valores não outliers
                # Renomeia de volta para o nome original da coluna
                .alias(col_name)
            )
        elif treatment.lower() == "null":
            # Substitui outliers por null
            df_processed = df_processed.with_columns(
                pl.when(is_outlier)
                # Substitui por null, mantendo o tipo
                .then(pl.lit(None, dtype=df_processed[col_name].dtype))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )

    return df_processed


def calculate_returns(
    data: pl.DataFrame,
    price_col: str = "close",
    log_returns: bool = True,
    output_col: str = "returns",
) -> pl.DataFrame:
    """
    Calcula os retornos (simples ou logarítmicos) para uma coluna de preço.

    Args:
        data (pl.DataFrame): DataFrame de entrada.
        price_col (str): Nome da coluna de preço a ser usada. Default: 'close'.
        log_returns (bool): Se True, calcula retornos logarítmicos. Se False,
                            calcula retornos simples. Default: True.
        output_col (str): Nome da coluna de saída para os retornos. Default: 'returns'.

    Returns:
        pl.DataFrame: DataFrame com a coluna de retornos adicionada.
    """
    if price_col not in data.columns:
        raise pl.exceptions.ColumnNotFoundError(
            f"Coluna de preço '{price_col}' não encontrada."
        )

    if log_returns:
        # Retorno logarítmico: ln(P_t / P_{t-1})
        returns_expr = pl.col(price_col).log().diff(1)
    else:
        # Retorno simples: (P_t / P_{t-1}) - 1
        returns_expr = pl.col(price_col).pct_change(1)

    return data.with_columns(returns_expr.alias(output_col))


def create_lagged_features(
    data: pl.DataFrame,
    columns: List[str],
    lags: Union[int, List[int]],
    time_col: Optional[str] = None,
    suffix_format: str = "_lag_{}",
) -> pl.DataFrame:
    """
    Cria features defasadas (lags) para colunas especificadas.

    Args:
        data (pl.DataFrame): DataFrame de entrada.
        columns (List[str]): Lista de nomes das colunas para criar lags.
        lags (Union[int, List[int]]): O número de períodos de lag ou uma lista
                                       de períodos de lag a serem criados.
        time_col (Optional[str]): Nome da coluna de tempo. Se fornecido, garante que
                                  os dados estejam ordenados por tempo antes de
                                  calcular os lags. Default: None.
        suffix_format (str): Formato para o sufixo das colunas defasadas. Deve conter
                             um placeholder {} para o número do lag.
                             Default: "_lag_{}".

    Returns:
        pl.DataFrame: DataFrame com as colunas defasadas adicionadas.
    """
    df_lagged = data
    if time_col:
        _validate_time_column(data, time_col)
        df_lagged = data.sort(time_col)

    if isinstance(lags, int):
        lag_periods = [lags]
    elif isinstance(lags, list):
        lag_periods = lags
    else:
        raise TypeError(
            "O parâmetro 'lags' deve ser um inteiro ou uma lista de inteiros."
        )

    lag_exprs = []
    for col in columns:
        if col not in data.columns:
            raise pl.exceptions.ColumnNotFoundError(f"Coluna '{col}' não encontrada.")
        for lag in lag_periods:
            if lag <= 0:
                raise ValueError("Os valores de lag devem ser inteiros positivos.")
            lag_exprs.append(
                pl.col(col).shift(lag).alias(f"{col}{suffix_format.format(lag)}")
            )

    return df_lagged.with_columns(lag_exprs)


def add_temporal_features(
    data: pl.DataFrame,
    time_col: str,
    features: List[str] = ["weekday", "hour", "day", "month", "year"],
) -> pl.DataFrame:
    """
    Adiciona colunas com features extraídas da coluna de tempo.

    Args:
        data (pl.DataFrame): DataFrame de entrada.
        time_col (str): Nome da coluna de tempo (deve ser tipo temporal).
        features (List[str]): Lista das features a serem extraídas.
                              Opções suportadas: 'weekday', 'hour', 'day', 'month',
                              'year', 'week', 'quarter', 'ordinal_day'.
                              Default: ['weekday', 'hour', 'day', 'month', 'year'].

    Returns:
        pl.DataFrame: DataFrame com as features temporais adicionadas.
    """
    _validate_time_column(data, time_col)

    temporal_exprs = []
    time_col_expr = pl.col(time_col).dt

    for feature in features:
        feature_lower = feature.lower()
        alias_name = f"{time_col}_{feature_lower}"
        if feature_lower == "weekday":
            temporal_exprs.append(time_col_expr.weekday().alias(alias_name))
        elif feature_lower == "hour":
            # Verifica se a coluna é Datetime antes de extrair hora
            if isinstance(data[time_col].dtype, Datetime):
                temporal_exprs.append(time_col_expr.hour().alias(alias_name))
            else:
                logger.warning(
                    f"Não é possível extrair 'hour' da coluna tipo Date '{time_col}'. Pulando feature."
                )
        elif feature_lower == "day":
            temporal_exprs.append(time_col_expr.day().alias(alias_name))
        elif feature_lower == "month":
            temporal_exprs.append(time_col_expr.month().alias(alias_name))
        elif feature_lower == "year":
            temporal_exprs.append(time_col_expr.year().alias(alias_name))
        elif feature_lower == "week":
            temporal_exprs.append(time_col_expr.week().alias(alias_name))
        elif feature_lower == "quarter":
            temporal_exprs.append(time_col_expr.quarter().alias(alias_name))
        elif feature_lower == "ordinal_day":
            temporal_exprs.append(time_col_expr.ordinal_day().alias(alias_name))
        else:
            logger.warning(f"Feature temporal '{feature}' não suportada. Pulando.")

    if not temporal_exprs:
        return data  # Retorna original se nenhuma feature válida foi solicitada

    return data.with_columns(temporal_exprs)


def split_in_out_sample(
    data: pl.DataFrame,
    time_col: str,
    split_date: Union[str, datetime.date, datetime.datetime],
    include_split_date_in_sample: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Divide um DataFrame em conjuntos in-sample (treino) e out-of-sample (teste)
    baseado em uma data de corte.

    Args:
        data (pl.DataFrame): DataFrame de entrada, esperado estar ordenado por tempo.
        time_col (str): Nome da coluna de tempo.
        split_date (Union[str, datetime.date, datetime.datetime]): A data/hora para
                                      dividir os dados. Pode ser uma string
                                      (ex: "YYYY-MM-DD" ou "YYYY-MM-DD HH:MM:SS")
                                      ou um objeto date/datetime.
        include_split_date_in_sample (bool): Se True, a data de corte é incluída no
                                             conjunto in-sample. Se False, é incluída
                                             no out-of-sample. Default: True.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Uma tupla contendo (in_sample_df, out_of_sample_df).
    """
    _validate_time_column(data, time_col)

    # Converte split_date para datetime.date se for string para consistência na comparação
    # Se a coluna for Datetime, é melhor converter split_date para datetime.
    # Se a coluna for Date, converter para date.
    col_dtype = data[time_col].dtype
    if isinstance(split_date, str):
        try:
            if col_dtype == Date:
                split_date_dt = datetime.datetime.strptime(
                    split_date, "%Y-%m-%d"
                ).date()
            else:  # Assumindo Datetime
                split_date_dt = datetime.datetime.fromisoformat(split_date)
        except ValueError:
            raise ValueError(
                f"Formato de split_date string '{split_date}' inválido. Use YYYY-MM-DD ou formato ISO."
            )
    elif isinstance(split_date, datetime.datetime) and col_dtype == Date:
        split_date_dt = split_date.date()
    elif isinstance(split_date, datetime.date) and col_dtype != Date:
        # Se a coluna é Datetime, e split_date é Date, converter split_date para Datetime (meia-noite)
        split_date_dt = datetime.datetime.combine(split_date, datetime.time.min)
    else:
        split_date_dt = split_date  # Já está no tipo correto ou compatível

    if include_split_date_in_sample:
        in_sample = data.filter(pl.col(time_col) <= split_date_dt)
        out_of_sample = data.filter(pl.col(time_col) > split_date_dt)
    else:
        in_sample = data.filter(pl.col(time_col) < split_date_dt)
        out_of_sample = data.filter(pl.col(time_col) >= split_date_dt)

    return in_sample, out_of_sample
