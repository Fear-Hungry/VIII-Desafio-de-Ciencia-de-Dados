"""
**Módulo de Carregamento e Processamento de Dados (`data_loader.loader`)**

Este módulo fornece uma classe `DataLoader` genérica para carregar, pré-processar,
calcular indicadores técnicos e preparar dados financeiros para análise ou
possivelmente para alimentar outros sistemas (não diretamente a `BacktestingEngine`).

Utiliza Polars para manipulação eficiente.

**Nota:** Este módulo parece ser independente da estrutura `DataHandler` usada
pela `BacktestingEngine` e pode ter dependências de um pacote `indicators`
que não está totalmente definido neste contexto.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import logging

# from .settings import (COLUMN_NAMES, EXPECTED_DTYPES_CONVERSION_DICT, REQUIRED_COLUMNS) # Removido
from .types import DataType, DataFrequency, DataSourceType

# Definições padrão que antes viriam de .settings
COLUMN_NAMES = {
    # Mapeamentos comuns de nomes de coluna para um padrão interno
    # Exemplo: "Data": "date", "Timestamp": "datetime", "Último": "close", etc.
    # Estes são apenas exemplos, ajuste conforme os nomes comuns nos seus CSVs.
    "date": "date",
    "datetime": "date", # Mapear para 'date' como nome interno padrão para tempo
    "timestamp": "date",
    "time": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    # Adicione outros mapeamentos comuns que você espera encontrar
}

EXPECTED_DTYPES_CONVERSION_DICT = {
    # Colunas e seus tipos Polars esperados após a conversão inicial
    # O pré-processamento tentará converter para estes tipos.
    "date": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Float64, # Ou pl.Int64 dependendo dos seus dados
}

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close"] # Colunas mínimas esperadas após o pré-processamento básico

# Configuração do logger
logger = logging.getLogger(__name__)

# Importar funções de indicadores e o novo Enum
# !!! ATENÇÃO: Dependência de um pacote/módulo 'indicators' externo !!!
try:
    # Alterado para importação absoluta do pacote indicators da raiz do projeto
    from indicators import momentum, moving_averages, volatility
    from indicators.types import Indicator # Importar o Enum IndicatorType como Indicator para compatibilidade local ou renomear no uso

    INDICATORS_AVAILABLE = True
except ImportError as e_import_ind:
    logger.warning(
        f"Módulo 'indicators' não encontrado ou incompleto (Erro: {e_import_ind}). "
        + "Funcionalidade `add_technical_indicators` estará desabilitada."
    )
    INDICATORS_AVAILABLE = False
    # Cria um Enum dummy se não existir para evitar erros de NameError
    from enum import Enum

    class Indicator(Enum): # Enum Dummy para compatibilidade
        # Adicione aqui os membros esperados pelo DataLoader se necessário
        # Ex: SMA_5 = "sma_5"
        # Ou deixe vazio se a lógica do DataLoader puder lidar com um Enum vazio
        # para a lista `default_indicators_enums`
        SMA_5 = "sma_5"
        SMA_10 = "sma_10"
        SMA_20 = "sma_20"
        SMA_50 = "sma_50"
        SMA_200 = "sma_200"
        EMA_12 = "ema_12"
        EMA_26 = "ema_26"
        MACD = "macd"
        RSI_14 = "rsi_14"
        BB_20_2 = "bb_20_2"
        pass


class DataLoader:
    """
    **Classe Genérica para Carregamento e Processamento de Dados Financeiros**

    Oferece funcionalidades para carregar dados de CSV ou Parquet, realizar
    pré-processamento comum (renomear colunas, converter tipos, tratar nulos,
    ordenar por data), adicionar indicadores técnicos (se o módulo `indicators`
    estiver disponível) e dividir dados para treino/teste.

    **Não implementa a interface `DataHandler` e não é diretamente usada
    pela `BacktestingEngine`.**
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Inicializa o carregador de dados.

        Args:
            cache_dir (str, optional): Diretório para armazenar/ler arquivos
                                       de cache (ex: Parquet). Default: None.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir and not self.cache_dir.exists():
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Diretório de cache criado: {self.cache_dir}")
            except Exception as e:
                logger.error(
                    f"Erro ao criar diretório de cache {self.cache_dir}: {e}")
                self.cache_dir = None  # Desabilita cache se não puder criar

    def _get_cache_path(self, filepath: str, preprocess_params: dict) -> Optional[Path]:
        """
        Gera o caminho do arquivo de cache Parquet baseado no arquivo de origem e nos parâmetros de preprocessamento.
        """
        if not self.cache_dir:
            return None
        base = Path(filepath).stem
        # Serializa os parâmetros para string e gera hash curto
        params_str = json.dumps(preprocess_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        cache_name = f"{base}_pre_{params_hash}.parquet"
        return self.cache_dir / cache_name

    def _is_cache_valid(self, source_path: str, cache_path: Path) -> bool:
        """
        Verifica se o cache está atualizado em relação ao arquivo-fonte.
        """
        if not cache_path.exists():
            return False
        try:
            source_mtime = os.path.getmtime(source_path)
            cache_mtime = os.path.getmtime(cache_path)
            return cache_mtime > source_mtime
        except Exception as e:
            logger.warning(f"Erro ao verificar validade do cache: {e}")
            return False

    def load_with_cache(
        self, filepath: str, preprocess_params: dict = None, **kwargs
    ) -> Optional[pl.DataFrame]:
        """
        Carrega dados de CSV (ou Parquet), usando cache inteligente após preprocessamento.
        Args:
            filepath (str): Caminho do arquivo CSV ou Parquet.
            preprocess_params (dict): Parâmetros do preprocessamento.
            **kwargs: Argumentos adicionais para pl.read_csv.
        Returns:
            Optional[pl.DataFrame]: DataFrame Polars carregado e pré-processado.
        """
        preprocess_params = preprocess_params or {}
        cache_path = self._get_cache_path(filepath, preprocess_params)
        if cache_path and self._is_cache_valid(filepath, cache_path):
            logger.info(f"Carregando dados do cache otimizado: {cache_path}")
            try:
                return pl.read_parquet(str(cache_path))
            except Exception as e:
                logger.warning(
                    f"Falha ao ler cache Parquet: {e}. Recarregando do original."
                )
        # Decide se é CSV ou Parquet
        if filepath.lower().endswith(".parquet"):
            df = self.load_from_parquet(filepath, **kwargs)
        else:
            df = self.load_from_csv(filepath, **kwargs)
        if df is None:
            return None
        df = self.preprocess(df, **preprocess_params)
        # Salva no cache
        if cache_path:
            self.save_to_parquet(df, str(cache_path))
        return df

    def load_from_csv(
        self,
        filepath: str,
        use_cache: bool = True,
        preprocess_params: dict = None,
        **kwargs,
    ) -> Optional[pl.DataFrame]:
        """
        Carrega dados de um arquivo CSV para um DataFrame Polars, com cache inteligente opcional.
        Args:
            filepath (str): Caminho do arquivo CSV.
            use_cache (bool): Se True, utiliza cache otimizado (default: True).
            preprocess_params (dict): Parâmetros do preprocessamento (default: None).
            **kwargs: Argumentos adicionais para `pl.read_csv`.
        Returns:
            Optional[pl.DataFrame]: DataFrame Polars com os dados carregados e pré-processados, ou None se erro.
        """
        preprocess_params = preprocess_params or {}
        if use_cache:
            cache_path = self._get_cache_path(filepath, preprocess_params)
            if cache_path and self._is_cache_valid(filepath, cache_path):
                logger.info(
                    f"Carregando dados do cache otimizado: {cache_path}")
                try:
                    return pl.read_parquet(str(cache_path))
                except Exception as e:
                    logger.warning(
                        f"Falha ao ler cache Parquet: {e}. Recarregando do original."
                    )
        logger.info(f"Carregando dados de {filepath}")
        try:
            df = pl.read_csv(source=filepath, **kwargs)
            logger.info(
                f"Dados carregados com sucesso de {filepath}. Dimensões: {df.shape}"
            )
            df = self.preprocess(
                df,
                use_cache=use_cache,
                preprocess_params=preprocess_params,
                source_path=filepath,
            )
            return df
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Erro ao carregar dados CSV de {filepath}: {e}")
            return None

    def load_from_parquet(
        self,
        filepath: str,
        use_cache: bool = True,
        preprocess_params: dict = None,
        **kwargs,
    ) -> Optional[pl.DataFrame]:
        """
        Carrega dados de um arquivo Parquet para um DataFrame Polars, com cache inteligente opcional.
        Args:
            filepath (str): Caminho do arquivo Parquet.
            use_cache (bool): Se True, utiliza cache otimizado (default: True).
            preprocess_params (dict): Parâmetros do preprocessamento (default: None).
            **kwargs: Argumentos adicionais para `pl.read_parquet`.
        Returns:
            Optional[pl.DataFrame]: DataFrame Polars com os dados carregados e pré-processados, ou None se erro.
        """
        preprocess_params = preprocess_params or {}
        if use_cache:
            cache_path = self._get_cache_path(filepath, preprocess_params)
            if cache_path and self._is_cache_valid(filepath, cache_path):
                logger.info(
                    f"Carregando dados do cache otimizado: {cache_path}")
                try:
                    return pl.read_parquet(str(cache_path))
                except Exception as e:
                    logger.warning(
                        f"Falha ao ler cache Parquet: {e}. Recarregando do original."
                    )
        logger.info(f"Carregando dados de {filepath}")
        try:
            df = pl.read_parquet(source=filepath, **kwargs)
            logger.info(
                f"Dados carregados com sucesso de {filepath}. Dimensões: {df.shape}"
            )
            df = self.preprocess(
                df,
                use_cache=use_cache,
                preprocess_params=preprocess_params,
                source_path=filepath,
            )
            return df
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Erro ao carregar dados Parquet de {filepath}: {e}")
            return None

    def fill_missing_values(
        self,
        df: pl.DataFrame,
        method: str = "ffill_bfill",
        limit: int = None,
        cols: list = None,
        spline_order: int = 3,
    ) -> pl.DataFrame:
        """
        Preenche valores ausentes nas colunas especificadas usando o método desejado.
        Args:
            df (pl.DataFrame): DataFrame de entrada.
            method (str): 'ffill', 'bfill', 'ffill_bfill', 'zeros', 'drop', 'interpolate', 'spline'.
            limit (int): Para ffill/bfill, máximo de barras consecutivas a preencher (None = sem limite).
            cols (list): Lista de colunas a processar (default: OHLCV).
            spline_order (int): Ordem do spline (usado se method='spline').
        Returns:
            pl.DataFrame: DataFrame com valores ausentes preenchidos.
        """
        import numpy as np

        try:
            from scipy.interpolate import UnivariateSpline
        except ImportError:
            UnivariateSpline = None

        if df.is_empty():
            return df
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        cols = cols or [col for col in ohlcv_cols if col in df.columns]
        if not cols:
            return df
        df_out = df
        for col in cols:
            if not df_out[col].is_null().any():
                continue
            if method == "drop":
                df_out = df_out.drop_nulls(subset=[col])
            elif method == "zeros":
                df_out = df_out.with_columns(
                    pl.col(col).fill_null(0.0).alias(col))
            elif method == "ffill":
                df_out = df_out.with_columns(
                    pl.col(col).fill_null(
                        strategy="forward", limit=limit).alias(col)
                )
            elif method == "bfill":
                df_out = df_out.with_columns(
                    pl.col(col).fill_null(
                        strategy="backward", limit=limit).alias(col)
                )
            elif method == "ffill_bfill":
                df_out = df_out.with_columns(
                    pl.col(col).fill_null(
                        strategy="forward", limit=limit).alias(col)
                )
                df_out = df_out.with_columns(
                    pl.col(col).fill_null(
                        strategy="backward", limit=limit).alias(col)
                )
            elif method == "interpolate":
                try:
                    df_out = df_out.with_columns(
                        pl.col(col).interpolate().alias(col))
                except Exception:
                    # fallback para versões antigas do polars
                    df_out = df_out.with_columns(
                        pl.col(col).fill_null(strategy="linear").alias(col)
                    )
            elif method == "spline" and UnivariateSpline is not None:
                # Spline só faz sentido para séries contínuas e com pelo menos spline_order+1 pontos
                arr = df_out[col].to_numpy()
                x = np.arange(len(arr))
                mask = ~np.isnan(arr)
                if mask.sum() > spline_order:
                    spline = UnivariateSpline(
                        x[mask], arr[mask], k=spline_order, s=0)
                    arr_interp = arr.copy()
                    arr_interp[~mask] = spline(x[~mask])
                    df_out = df_out.with_columns(pl.Series(col, arr_interp))
            else:
                # fallback: só faz ffill_bfill
                df_out = df_out.with_columns(
                    pl.col(col).fill_null(strategy="forward").alias(col)
                )
                df_out = df_out.with_columns(
                    pl.col(col).fill_null(strategy="backward").alias(col)
                )
        return df_out

    def preprocess(
        self,
        df: pl.DataFrame,
        time_col_priority: List[str] = ["date", "datetime", "timestamp"],
        use_cache: bool = True,
        preprocess_params: dict = None,
        source_path: str = None,
    ) -> pl.DataFrame:
        """
        Realiza pré-processamento básico em um DataFrame Polars, salvando no cache se configurado.
        Args:
            df (pl.DataFrame): DataFrame Polars a ser pré-processado.
            time_col_priority (List[str]): Lista ordenada de nomes de coluna a procurar para identificar a coluna de tempo.
            use_cache (bool): Se True, salva o resultado no cache (default: True).
            preprocess_params (dict): Parâmetros do preprocessamento (default: None).
            source_path (str): Caminho do arquivo-fonte original (necessário para cache).
        Returns:
            pl.DataFrame: DataFrame Polars pré-processado.
        """
        if not isinstance(df, pl.DataFrame):
            logger.error("Input para preprocess não é um DataFrame Polars.")
            return df  # Retorna o input original se não for DF

        logger.info("Iniciando pré-processamento...")
        original_shape = df.shape

        # Renomear colunas para minúsculas
        df = df.rename({col: col.lower() for col in df.columns})
        logger.debug(f"Colunas após renomear: {df.columns}")

        # Identificar e converter coluna de data/hora
        date_col = None
        for col_name in time_col_priority:
            if col_name in df.columns:
                logger.debug(
                    f"Coluna de tempo potencial encontrada: '{col_name}'. Tentando converter..."
                )
                try:
                    if df[col_name].dtype == pl.Utf8:
                        # Tenta converter de string para Datetime (mais geral)
                        df = df.with_columns(
                            pl.col(col_name).str.to_datetime(
                                strict=False, time_unit="us"
                            )
                        )
                        logger.info(
                            f"Coluna '{col_name}' convertida para Datetime com unidade 'us'."
                        )
                        break  # Sai do loop de time_unit se a conversão for bem-sucedida
                    elif df[col_name].dtype == pl.Date:
                        pass
                    elif df[col_name].dtype == pl.Datetime:
                        pass
                    # Adicione outras conversões se necessário (ex: Int -> Datetime)

                    # Se a conversão funcionou ou já era temporal, define como date_col
                    col_dtype = df[col_name].dtype
                    # Adicione pl.Duration à lista se for um tipo temporal relevante para seus dados
                    is_known_temporal = col_dtype in [
                        pl.Datetime, pl.Date, pl.Time]

                    if hasattr(col_dtype, 'is_temporal') and col_dtype.is_temporal():
                        date_col = col_name
                        logger.info(
                            f"Coluna de tempo (via is_temporal) identificada e processada: '{date_col}' ({col_dtype})"
                        )
                        break  # Para na primeira coluna de tempo encontrada e processada
                    elif is_known_temporal:
                        date_col = col_name
                        logger.info(
                            f"Coluna de tempo (via tipo conhecido) identificada e processada: '{date_col}' ({col_dtype})"
                        )
                        break  # Para na primeira coluna de tempo encontrada e processada
                    else:
                        logger.warning(
                            f"Falha ao converter '{col_name}' para tipo temporal ou tipo não suportado: {col_dtype}"
                        )

                except Exception as e_time:
                    logger.warning(
                        f"Erro ao tentar converter coluna '{col_name}' como temporal: {e_time}"
                    )
                    continue  # Tenta a próxima coluna na prioridade
        if not date_col:
            logger.warning(
                "Nenhuma coluna de tempo válida encontrada ou convertida com sucesso."
            )

        # Ordenar pelo campo de data/hora, se existir e for válido
        if date_col:
            logger.debug(f"Ordenando DataFrame pela coluna: {date_col}")
            df = df.sort(date_col)
            # Remove duplicatas de tempo, mantendo a última ocorrência
            initial_rows_sort = df.height
            df = df.unique(subset=[date_col], keep="last", maintain_order=True)
            if initial_rows_sort > df.height:
                logger.info(
                    f"{initial_rows_sort - df.height} linhas duplicadas removidas com base em '{date_col}'."
                )
        else:
            logger.warning(
                "Não foi possível ordenar o DataFrame por data/hora.")

        # Garantir que colunas OHLCV sejam numéricas (Float64)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            if col in df.columns:
                if df[col].dtype != pl.Float64:
                    logger.debug(f"Convertendo coluna '{col}' para Float64.")
                    try:
                        df = df.with_columns(
                            pl.col(col).cast(pl.Float64, strict=False))
                    except Exception as e_cast:
                        logger.error(
                            f"Erro ao converter '{col}' para Float64: {e_cast}. Coluna pode conter valores não numéricos."
                        )
                        # Poderia tentar limpar a coluna antes do cast ou remover

        # Remover linhas com valores ausentes em colunas críticas (OHLCV), APÓS cast
        cols_to_check_nulls = [
            col
            for col in ohlcv_cols
            if col in df.columns and df[col].dtype == pl.Float64
        ]
        if cols_to_check_nulls:
            # NOVO: aplicar método de preenchimento customizado se passado
            fill_method = None
            fill_limit = None
            fill_spline_order = 3
            if preprocess_params:
                fill_method = preprocess_params.get("fill_method")
                fill_limit = preprocess_params.get("fill_limit")
                fill_spline_order = preprocess_params.get(
                    "fill_spline_order", 3)
            if fill_method:
                df = self.fill_missing_values(
                    df,
                    method=fill_method,
                    limit=fill_limit,
                    cols=cols_to_check_nulls,
                    spline_order=fill_spline_order,
                )
            else:
                initial_rows_null = df.height
                df = df.drop_nulls(subset=cols_to_check_nulls)
                removed_rows_null = initial_rows_null - df.height
                if removed_rows_null > 0:
                    logger.info(
                        f"{removed_rows_null} linhas removidas devido a valores nulos em {cols_to_check_nulls}."
                    )
        else:
            logger.warning(
                "Nenhuma coluna OHLCV numérica encontrada para verificação de nulos."
            )

        final_shape = df.shape
        logger.info(
            f"Pré-processamento concluído. Dimensões: {original_shape} -> {final_shape}"
        )

        # Salva no cache se habilitado
        if use_cache and self.cache_dir and source_path:
            cache_path = self._get_cache_path(
                source_path,
                preprocess_params or {"time_col_priority": time_col_priority},
            )
            if cache_path:
                self.save_to_parquet(df, str(cache_path))
        return df

    def add_technical_indicators(
        self, df: pl.DataFrame, indicators_to_add: Optional[List[Indicator]] = None
    ) -> pl.DataFrame:
        """
        Adiciona indicadores técnicos especificados ao DataFrame Polars.

        **Depende do módulo `indicators` externo.** Se não estiver disponível,
        nenhum indicador será adicionado.

        Args:
            df (pl.DataFrame): DataFrame Polars com dados financeiros pré-processados
                               (espera colunas como 'close', 'high', 'low', 'volume').
            indicators_to_add (Optional[List[Indicator]]): Lista de Enum `Indicator`
                (definido em `indicators.types`) a serem adicionados.
                Se None, calcula um conjunto padrão (SMA, EMA, MACD, RSI, BB).

        Returns:
            pl.DataFrame: DataFrame Polars com colunas de indicadores adicionadas,
                        ou o DataFrame original se `indicators` não estiver disponível
                        ou ocorrer um erro.
        """
        if not INDICATORS_AVAILABLE:
            logger.warning(
                "Módulo 'indicators' não disponível. Pulando adição de indicadores."
            )
            return df

        if not isinstance(df, pl.DataFrame) or df.is_empty():
            logger.warning(
                "DataFrame inválido ou vazio fornecido para add_technical_indicators."
            )
            return df

        # Verifica se as colunas necessárias para os indicadores existem
        required_for_indicators = ["close"]  # Mínimo necessário
        # Adicionar outras como high, low, volume se os indicadores as usarem
        if "high" in df.columns and "low" in df.columns:
            required_for_indicators.extend(["high", "low"])
        if "volume" in df.columns:
            required_for_indicators.append("volume")

        missing_req = [
            col
            for col in required_for_indicators
            if col not in df.columns or not pl.datatypes.is_numeric(df[col].dtype)
        ]
        if missing_req:
            logger.warning(
                f"Colunas numéricas necessárias ({missing_req}) para indicadores ausentes. Pulando."
            )
            return df

        # Define conjunto padrão se nenhum for fornecido
        if indicators_to_add is None:
            # Usando nomes do Enum importado (se existir)
            default_indicators_enums = [
                Indicator.SMA_5,
                Indicator.SMA_10,
                Indicator.SMA_20,
                Indicator.SMA_50,
                Indicator.SMA_200,
                Indicator.EMA_12,
                Indicator.EMA_26,
                Indicator.MACD,
                Indicator.RSI_14,
                Indicator.BB_20_2,
            ]
            # Filtra caso o Enum não tenha todos esses valores
            indicators_to_add = [
                ind for ind in default_indicators_enums if isinstance(ind, Indicator)
            ]
            if not indicators_to_add:
                logger.warning(
                    "Enum Indicator vazio ou não contém valores padrão. Nenhum indicador será adicionado."
                )
                return df

        logger.info(
            f"Adicionando indicadores técnicos: {[ind.name for ind in indicators_to_add]}"
        )
        df_with_indicators = df  # Começa com o DF original
        processed_indicators = set()

        # Mapeamento de Enum para função e argumentos (exemplo)
        # Isso pode ser mais robusto, talvez usando um padrão de fábrica ou registro
        indicator_map = {
            Indicator.SMA_5: (moving_averages.add_sma, {"period": 5}),
            Indicator.SMA_10: (moving_averages.add_sma, {"period": 10}),
            Indicator.SMA_20: (moving_averages.add_sma, {"period": 20}),
            Indicator.SMA_50: (moving_averages.add_sma, {"period": 50}),
            Indicator.SMA_200: (moving_averages.add_sma, {"period": 200}),
            Indicator.EMA_12: (moving_averages.add_ema, {"span": 12}),
            Indicator.EMA_26: (moving_averages.add_ema, {"span": 26}),
            Indicator.MACD: (momentum.add_macd, {}),  # Assume padrão 12, 26, 9
            Indicator.RSI_14: (momentum.add_rsi, {"period": 14}),
            Indicator.BB_20_2: (
                volatility.add_bollinger_bands,
                {"period": 20, "std_dev": 2},
            ),
        }

        for indicator_enum in indicators_to_add:
            if (
                indicator_enum in processed_indicators
                or indicator_enum not in indicator_map
            ):
                if indicator_enum not in indicator_map:
                    logger.warning(
                        f"Indicador {indicator_enum.name} não mapeado para função. Pulando."
                    )
                continue

            func, kwargs = indicator_map[indicator_enum]
            logger.debug(
                f"Calculando {indicator_enum.name} com kwargs: {kwargs}")
            try:
                df_with_indicators = func(df_with_indicators, **kwargs)
                processed_indicators.add(indicator_enum)
                logger.debug(f"{indicator_enum.name} adicionado com sucesso.")

                # Marca dependências como processadas (ex: MACD calcula EMAs)
                if indicator_enum == Indicator.MACD:
                    if Indicator.EMA_12 in indicator_map:
                        processed_indicators.add(Indicator.EMA_12)
                    if Indicator.EMA_26 in indicator_map:
                        processed_indicators.add(Indicator.EMA_26)
                # Adicione outras dependências se necessário (ex: BB usa SMA)
                if (
                    indicator_enum == Indicator.BB_20_2
                    and Indicator.SMA_20 in indicator_map
                ):
                    processed_indicators.add(Indicator.SMA_20)

            except Exception as e_ind:
                logger.error(
                    f"Erro ao calcular indicador {indicator_enum.name}: {e_ind}"
                )
                # Continua para o próximo indicador

        final_indicator_cols = [
            col for col in df_with_indicators.columns if col not in df.columns
        ]
        logger.info(f"Indicadores adicionados: {final_indicator_cols}")
        logger.info(f"Dimensões após indicadores: {df_with_indicators.shape}")
        return df_with_indicators

    def split_train_test(
        self, df: pl.DataFrame, train_ratio: float = 0.8, date_col: Optional[str] = None
    ) -> Optional[Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Divide o DataFrame em conjuntos de treino e teste baseado em proporção temporal.

        Args:
            df (pl.DataFrame): DataFrame Polars ordenado por tempo.
            train_ratio (float): Proporção do dataset a ser usada para treino (0 a 1).
                                 Padrão: 0.8 (80%).
            date_col (Optional[str]): Nome da coluna de data/hora para usar na divisão.
                                      Se None, tenta encontrar automaticamente.

        Returns:
            Optional[Tuple[pl.DataFrame, pl.DataFrame]]: Tupla contendo (df_train, df_test),
                                                         ou None se a divisão falhar.
        """
        if not isinstance(df, pl.DataFrame) or df.is_empty():
            logger.error("DataFrame inválido ou vazio para split_train_test.")
            return None
        if not (0 < train_ratio < 1):
            logger.error("train_ratio deve estar entre 0 e 1 (exclusivo).")
            return None

        # Encontra coluna de data se não especificada
        if date_col is None:
            potential_cols = [
                c
                for c in ["date", "datetime", "timestamp"]
                if c in df.columns and pl.datatypes.is_temporal(df[c].dtype)
            ]
            if not potential_cols:
                logger.error(
                    "Nenhuma coluna temporal encontrada para divisão. Especifique `date_col`."
                )
                return None
            date_col = potential_cols[0]
            logger.info(f"Coluna temporal para divisão: '{date_col}'.")
        elif date_col not in df.columns or not pl.datatypes.is_temporal(
            df[date_col].dtype
        ):
            logger.error(
                f"Coluna '{date_col}' especificada não é válida ou não é temporal."
            )
            return None

        # Garante ordenação (redundante se preprocess foi chamado, mas seguro)
        df = df.sort(date_col)

        n_rows = df.height
        n_train = int(n_rows * train_ratio)
        n_test = n_rows - n_train

        if n_train == 0 or n_test == 0:
            logger.error(
                f"Divisão resultou em conjunto de treino ({n_train}) ou teste ({n_test}) vazio."
            )
            return None

        df_train = df.slice(offset=0, length=n_train)
        df_test = df.slice(offset=n_train, length=n_test)

        logger.info(
            f"Dados divididos: Treino={df_train.shape}, Teste={df_test.shape}")
        return df_train, df_test

    def create_features_target(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        lookahead: int = 1,
        binary_threshold: Optional[float] = None,
    ) -> Optional[Tuple[pl.DataFrame, pl.Series]]:
        """
        Cria DataFrames de features e target (Series) para modelos de ML.

        Args:
            df (pl.DataFrame): DataFrame com dados e indicadores.
            feature_cols (List[str]): Lista de nomes de colunas a serem usadas como features.
            target_col (str): Nome da coluna a ser usada para criar o target
                              (normalmente 'close').
            lookahead (int): Número de períodos no futuro para prever a mudança.
                             (Padrão: 1). O target será a mudança percentual
                             de `target_col` entre t e t+`lookahead`.
            binary_threshold (Optional[float]): Se fornecido, converte o target
                (mudança percentual) em binário: 1 se a mudança >= threshold,
                0 caso contrário. Se None, o target é a mudança percentual real.

        Returns:
            Optional[Tuple[pl.DataFrame, pl.Series]]: Tupla (features, target), ou None se erro.
                                                     Features e target são alinhados,
                                                     removendo as últimas `lookahead` linhas
                                                     onde o target não pode ser calculado.
        """
        if not isinstance(df, pl.DataFrame) or df.is_empty():
            logger.error(
                "DataFrame inválido ou vazio para create_features_target.")
            return None
        if (
            not feature_cols
            or target_col not in df.columns
            or not all(fc in df.columns for fc in feature_cols)
        ):
            logger.error("Colunas de features ou target inválidas/ausentes.")
            return None
        if lookahead < 1:
            logger.error("lookahead deve ser >= 1.")
            return None

        logger.info(f"Criando features e target (lookahead={lookahead})...")

        # Calcula a mudança futura (target)
        df = df.with_columns(
            (pl.col(target_col).shift(-lookahead) / pl.col(target_col) - 1).alias(
                "target_pct_change"
            )
        )

        # Remove as últimas N linhas que terão target nulo
        df_features_target = df.slice(offset=0, length=df.height - lookahead)

        # Define o target final
        if binary_threshold is not None:
            logger.info(
                f"Convertendo target para binário (threshold={binary_threshold})"
            )
            target = (df_features_target["target_pct_change"] >= binary_threshold).cast(
                pl.Int8
            )
            target.rename("target_binary")
        else:
            target = df_features_target["target_pct_change"]
            target.rename("target")

        # Seleciona as features
        features = df_features_target.select(feature_cols)

        # Remove quaisquer linhas restantes com NaNs nas features ou target
        combined = pl.concat([features, target.to_frame()], how="horizontal")
        combined_clean = combined.drop_nulls()
        rows_removed = combined.height - combined_clean.height
        if rows_removed > 0:
            logger.info(
                f"{rows_removed} linhas removidas devido a NaNs nas features/target finais."
            )

        if combined_clean.is_empty():
            logger.error(
                "Nenhuma linha restante após criação de features/target e limpeza de NaNs."
            )
            return None

        final_features = combined_clean.select(feature_cols)
        # Usa o nome que foi dado ao target
        final_target = combined_clean[target.name]

        logger.info(
            f"Features criadas: {final_features.shape}, Target criado: ({final_target.len()})"
        )
        return final_features, final_target

    def save_to_parquet(self, df: pl.DataFrame, filepath: str, **kwargs) -> bool:
        """
        Salva um DataFrame Polars em formato Parquet.

        Args:
            df (pl.DataFrame): DataFrame a ser salvo.
            filepath (str): Caminho do arquivo Parquet de destino.
            **kwargs: Argumentos adicionais para `df.write_parquet` (ex: compression='snappy').

        Returns:
            bool: True se salvo com sucesso, False caso contrário.
        """
        if not isinstance(df, pl.DataFrame):
            logger.error(
                "Input para save_to_parquet não é um DataFrame Polars.")
            return False

        logger.info(f"Salvando DataFrame ({df.shape}) para {filepath}")
        try:
            # Garante que o diretório exista
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(file=filepath, **kwargs)
            logger.info(f"DataFrame salvo com sucesso em {filepath}")
            return True
        except Exception as e:
            logger.error(
                f"Erro ao salvar DataFrame em Parquet ({filepath}): {e}")
            return False
