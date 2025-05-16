import datetime
from typing import Any, Dict, List, Optional  # Adicionado Any e Dict

import polars as pl
import logging
from events import MarketEvent  # Ajuste o caminho se necessário

from .base import DataHandler

# Supondo que você tenha uma biblioteca de banco de dados, ex: sqlalchemy, pyodbc, psycopg2
# import sqlalchemy


logger = logging.getLogger(__name__)


class SQLDataHandler(DataHandler):
    """
    Manipulador de dados que busca dados históricos de um banco de dados SQL.

    Esta classe implementa a interface DataHandler, conectando-se a um
    banco de dados SQL para fornecer barras de mercado para a engine de backtesting.
    """

    def __init__(
        self,
        db_connection_details: Any,  # Ex: string de conexão, objeto de conexão
        symbol_list: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        # Adicione outros parâmetros necessários (nome da tabela, schema, etc.)
        table_name: str,
        time_column: str = "timestamp",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        symbol_col: str = "symbol",
        # Método de preenchimento de valores ausentes
        fill_missing_method: str = "ffill_bfill",
        # Se True, valida e corrige inconsistências OHLC
        validate_ohlc: bool = True,
        # Configuração de detecção de outliers
        outlier_detection: Optional[Dict[str, Any]] = None,
    ):
        """
        Inicializa o SQLDataHandler.

        Args:
            db_connection_details: Informações para conectar ao banco de dados.
            symbol_list: Lista de símbolos a serem carregados.
            start_date: Data de início para buscar dados.
            end_date: Data de fim para buscar dados.
            table_name: Nome da tabela que contém os dados OHLCV.
            time_column: Nome da coluna de timestamp no banco.
            open_col, high_col, low_col, close_col, volume_col: Nomes das colunas de dados.
            symbol_col: Nome da coluna que identifica o símbolo no banco.
            fill_missing_method (str): Método para preencher valores ausentes:
                                      - 'ffill_bfill': Forward fill seguido de backward fill (padrão)
                                      - 'interpolate': Interpolação linear
                                      - 'prev_day': Usar valores do dia anterior
                                      - 'zeros': Preencher com zeros
                                      - 'drop': Remover linhas com valores ausentes
            validate_ohlc (bool): Se True (padrão), valida e corrige inconsistências nos dados OHLC
            outlier_detection (Optional[Dict[str, Any]]): Configurações para detecção de outliers:
                                      - 'method': 'std_dev', 'iqr', 'absolute' ou None para desativado
                                      - 'threshold': limiar para o método (ex: 3 para std_dev, 1.5 para IQR)
                                      - 'action': 'clip', 'remove', ou 'ignore'
                                      - 'columns': lista de colunas para aplicar, padrão ['open','high','low','close','volume']
        """
        self._db_connection = self._connect_db(db_connection_details)
        self._symbol_list = symbol_list
        self._start_date = start_date
        self._end_date = end_date
        self._table_name = table_name
        self._time_col = time_column
        # Corrigido para usar os nomes das colunas passados
        self._ohlcv_cols = [open_col, high_col, low_col, close_col, volume_col]
        self._ohlcv_cols_map = {  # Mapa para renomear se necessário
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
            volume_col: "volume",
        }
        self._symbol_col = symbol_col
        self._fill_missing_method = fill_missing_method
        self._validate_ohlc = validate_ohlc

        # Configuração padrão para detecção de outliers se não fornecida
        self._outlier_detection = outlier_detection or {
            "method": None,  # Sem detecção de outliers por padrão
            "threshold": 3.0,  # Para std_dev
            "action": "clip",  # Limitar aos valores limite
            "columns": ["open", "high", "low", "close", "volume"],
        }

        # Validar configuração de outliers
        if self._outlier_detection.get("method") not in [
            None,
            "std_dev",
            "iqr",
            "absolute",
        ]:
            logger.warning(
                f"Método de detecção de outliers '{self._outlier_detection.get('method')}' inválido. Desativando."
            )
            self._outlier_detection["method"] = None

        # Cache para dados recentes
        self._latest_symbol_data: Dict[str, pl.DataFrame] = {}
        # Todos os timestamps únicos
        self._current_timestamps: Optional[pl.Series] = self._load_all_timestamps()
        self._timestamp_index: int = 0  # Índice para o timestamp atual

        # Inicialmente, o backtest pode continuar se houver timestamps
        self._continue_backtest = (
            self._current_timestamps is not None and len(self._current_timestamps) > 0
        )

        if self._current_timestamps is not None:
            logger.info(
                f"SQLDataHandler inicializado. {len(self._current_timestamps)} timestamps carregados entre {self._start_date} e {self._end_date}."
            )
        else:
            logger.warning(
                "SQLDataHandler inicializado, mas não foi possível carregar timestamps."
            )

    def _connect_db(self, details: Any) -> Any:
        """Método auxiliar para estabelecer a conexão com o BD."""
        logger.info("Conectando ao banco de dados...")
        # Implementar lógica de conexão aqui (ex: usando SQLAlchemy)
        # Exemplo:
        # try:
        #   engine = sqlalchemy.create_engine(details)
        #   connection = engine.connect()
        #   logger.info("Conexão com o banco de dados estabelecida.")
        #   return connection
        # except Exception as e:
        #   logger.error(f"Erro ao conectar ao banco de dados: {e}")
        #   raise # Relança a exceção para indicar falha na inicialização
        raise NotImplementedError(
            "A lógica de conexão com o banco de dados precisa ser implementada."
        )
        # return None # Placeholder

    def _load_all_timestamps(self) -> Optional[pl.Series]:
        """Carrega todos os timestamps únicos e ordenados do período."""
        if not self._db_connection:
            logger.error("Sem conexão com o banco de dados para carregar timestamps.")
            return None
        logger.info(
            f"Carregando timestamps únicos de {self._start_date} a {self._end_date}..."
        )
        try:
            # Construir a query SQL para buscar timestamps únicos
            # Garantir que a query use placeholders adequados para o seu driver de BD
            query = f"""
            SELECT DISTINCT "{self._time_col}"
            FROM "{self._table_name}"
            WHERE "{self._time_col}" >= ? AND "{self._time_col}" <= ?
            ORDER BY "{self._time_col}" ASC;
            """  # Adicionado aspas duplas para compatibilidade com alguns SQL dialects

            # Executar a query usando a conexão (_db_connection)
            # Substitua pela sua função de execução de query que retorna um DataFrame Polars
            # Ex: df_times = pl.read_database(query=query, connection=self._db_connection, params=(self._start_date, self._end_date))

            # Placeholder - Substitua pela execução real da query
            logger.debug(
                f"Executando query de timestamps: {query} com params ({self._start_date}, {self._end_date})"
            )
            # timestamps = self._execute_query(query, params=(self._start_date, self._end_date))['timestamp_column_name']
            raise NotImplementedError(
                "A lógica para buscar timestamps do banco precisa ser implementada."
            )
            # return timestamps # Placeholder

        except Exception as e:
            logger.error(f"Erro ao carregar timestamps do banco de dados: {e}")
            return None

    def _get_data_for_timestamp(
        self, timestamp: datetime.datetime
    ) -> Dict[str, pl.DataFrame]:
        """Busca dados para todos os símbolos em um timestamp específico."""
        if not self._db_connection:
            logger.error(
                f"Sem conexão com o banco para buscar dados no timestamp {timestamp}."
            )
            return {}

        data_for_timestamp = {}
        # Colunas a selecionar do banco, incluindo a de símbolo
        db_cols_to_select = [self._symbol_col, self._time_col] + self._ohlcv_cols
        # Usa nomes originais do DB
        select_clause = ", ".join([f'"{col}"' for col in db_cols_to_select])
        # Placeholders para a cláusula IN
        in_placeholders = ",".join("?" * len(self._symbol_list))

        try:
            # Query para buscar dados de todos os símbolos no timestamp atual
            query = f"""
            SELECT {select_clause}
            FROM "{self._table_name}"
            WHERE "{self._time_col}" = ? AND "{self._symbol_col}" IN ({in_placeholders});
            """
            params = [timestamp] + self._symbol_list

            # Executar a query -> Polars DataFrame
            # df_pl = pl.read_database(query=query, connection=self._db_connection, params=params)
            logger.debug(f"Executando query de dados: {query} com params ({params})")
            # Placeholder
            raise NotImplementedError(
                "Lógica para buscar dados por timestamp do banco precisa ser implementada."
            )
            # df_pl = self._execute_query_to_polars(query, params=params)

            # Organizar dados por símbolo e renomear colunas
            # for symbol in self._symbol_list:
            #     symbol_data = df_pl.filter(pl.col(self._symbol_col) == symbol)
            #     if not symbol_data.is_empty():
            #         # Renomeia as colunas do DB para o padrão ('open', 'high', etc.)
            #         renamed_data = symbol_data.rename(self._ohlcv_cols_map)
            #         # Seleciona apenas as colunas padrão + time_col
            #         data_for_timestamp[symbol] = renamed_data.select(self._time_col, 'open', 'high', 'low', 'close', 'volume')

        except Exception as e:
            logger.error(f"Erro ao buscar dados para o timestamp {timestamp}: {e}")

        return data_for_timestamp  # Pode conter dados para alguns símbolos ou nenhum

    @property
    def symbols(self) -> List[str]:
        """Retorna a lista de símbolos gerenciados."""
        return self._symbol_list

    @property
    def continue_backtest(self) -> bool:
        """Indica se ainda há timestamps a processar."""
        return self._continue_backtest

    def get_latest_bars_df(self, symbol: str, N: int = 1) -> Optional[pl.DataFrame]:
        """Retorna as N barras mais recentes para um símbolo."""
        # Implementação atual depende do cache _latest_symbol_data, que só guarda a última barra.
        # Uma implementação mais robusta buscaria do banco se N > 1 e os dados não estiverem cacheados.
        if symbol not in self._latest_symbol_data:
            # Tenta buscar a barra mais recente se não estiver no cache (pode acontecer no primeiro passo)
            current_dt = (
                self._current_timestamps[self._timestamp_index - 1]
                if self._timestamp_index > 0
                else None
            )
            if current_dt:
                logger.debug(
                    f"Cache vazio para {symbol}. Tentando buscar dados do timestamp {current_dt} no banco."
                )
                data_ts = self._get_data_for_timestamp(current_dt)
                if symbol in data_ts:
                    # Atualiza cache
                    self._latest_symbol_data[symbol] = data_ts[symbol]
                else:
                    logger.debug(
                        f"Nenhum dado encontrado para {symbol} no timestamp {current_dt} durante busca para get_latest_bars_df."
                    )
                    return None
            else:
                logger.debug(
                    f"Cache vazio para {symbol} e nenhum timestamp processado ainda."
                )
                return None  # Não há dados ainda

        if N == 1:
            # Retorna o DF da última barra (do cache)
            return self._latest_symbol_data.get(symbol)
        else:
            # Lógica para buscar N barras (requer query ao banco)
            logger.warning(
                f"Aviso: Buscando N={N} barras do banco para {symbol} (pode ser ineficiente)."
            )
            current_dt = self.get_latest_bar_datetime(
                symbol
            )  # Pega o timestamp da barra no cache
            if not current_dt:
                return None
            if not self._db_connection:
                logger.error(
                    f"Erro: Sem conexão com o banco para buscar {N} barras para {symbol}."
                )
                return None

            try:
                # Query para buscar as N últimas barras até o timestamp atual
                # Usar window function ou subquery dependendo do dialeto SQL
                query = f"""
                SELECT "{self._time_col}", {', '.join([f'"{c}"' for c in self._ohlcv_cols])}
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY "{self._symbol_col}" ORDER BY "{self._time_col}" DESC) as rn
                    FROM "{self._table_name}"
                    WHERE "{self._symbol_col}" = ? AND "{self._time_col}" <= ?
                ) tmp
                WHERE rn <= ?
                ORDER BY "{self._time_col}" ASC;
                """
                params = (symbol, current_dt, N)
                logger.debug(f"Executando query N barras: {query} com params {params}")

                # df_n_bars = pl.read_database(query=query, connection=self._db_connection, params=params)
                raise NotImplementedError(
                    "Lógica de busca de N barras do banco não implementada."
                )
                # Renomear colunas para o padrão
                # df_n_bars_renamed = df_n_bars.rename(self._ohlcv_cols_map)
                # return df_n_bars_renamed.select(self._time_col, 'open', 'high', 'low', 'close', 'volume')

            except Exception as e:
                logger.error(f"Erro ao buscar N={N} barras para {symbol}: {e}")
                return None

    def get_latest_bars(self, symbol: str, N: int = 1) -> Optional[pl.DataFrame]:
        """Alias para get_latest_bars_df."""
        return self.get_latest_bars_df(symbol, N)

    def get_latest_bar_value(self, symbol: str, val_type: str) -> Optional[float]:
        """Retorna um valor específico da barra mais recente (do cache)."""
        # Usa o cache _latest_symbol_data
        latest_bar = self._latest_symbol_data.get(symbol)
        # val_type deve ser um dos nomes padrão: 'open', 'high', 'low', 'close', 'volume'
        val_type_lower = val_type.lower()

        if (
            latest_bar is not None
            and not latest_bar.is_empty()
            and val_type_lower in latest_bar.columns
        ):
            try:
                # Pega o valor da primeira (e única) linha do DataFrame no cache
                value = latest_bar[val_type_lower][0]
                return float(value) if value is not None else None
            except (
                pl.exceptions.ColumnNotFoundError,
                IndexError,
                TypeError,
                ValueError,
            ):
                logger.error(
                    f"Erro ao extrair '{val_type_lower}' do cache para {symbol}."
                )
                return None
        # Se não estiver no cache, poderia tentar buscar diretamente do banco, mas por simplicidade não faremos isso aqui.
        # logger.debug(f"Valor '{val_type}' não encontrado no cache para {symbol}.")
        return None

    def get_latest_bar_datetime(self, symbol: str) -> Optional[datetime.datetime]:
        """Retorna o timestamp da barra mais recente (do cache)."""
        latest_bar = self._latest_symbol_data.get(symbol)
        if latest_bar is not None and not latest_bar.is_empty():
            try:
                # A coluna de tempo no cache deve ser a definida em self._time_col
                timestamp = latest_bar[self._time_col][0]
                return timestamp  # Assume que já é datetime
            except (pl.exceptions.ColumnNotFoundError, IndexError):
                logger.error(f"Erro ao extrair timestamp do cache para {symbol}.")
                return None
        # logger.debug(f"Timestamp não encontrado no cache para {symbol}.")
        return None

    def update_bars(self) -> Optional[MarketEvent]:
        """
        Avança para o próximo timestamp e busca os dados correspondentes do banco.
        """
        if self._current_timestamps is None or self._timestamp_index >= len(
            self._current_timestamps
        ):
            if self._continue_backtest:  # Imprime apenas uma vez
                logger.info("Fim dos timestamps alcançado.")
            self._continue_backtest = False
            return None  # Sinaliza fim do backtest

        # Pega o timestamp atual
        current_timestamp = self._current_timestamps[self._timestamp_index]
        logger.debug(
            f"update_bars: Processando timestamp {self._timestamp_index + 1}/{len(self._current_timestamps)}: {current_timestamp}"
        )

        # Busca dados do banco para este timestamp e atualiza o cache _latest_symbol_data
        latest_data = self._get_data_for_timestamp(current_timestamp)

        # Validar e corrigir dados para cada símbolo
        for symbol, df in latest_data.items():
            # Validar e corrigir dados OHLC para cada símbolo, se necessário
            if self._validate_ohlc and not df.is_empty():
                latest_data[symbol] = self._validate_ohlc_integrity(df, symbol)

            # Tratar valores ausentes, se existirem
            if not df.is_empty():
                null_exists = False
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns and df[col].is_null().any():
                        null_exists = True
                        break

                if null_exists:
                    latest_data[symbol] = self._fill_missing_values(
                        df, symbol, method=self._fill_missing_method
                    )

                # Tratar outliers, se configurado
                if self._outlier_detection.get("method") is not None:
                    latest_data[symbol] = self._handle_outliers(
                        latest_data[symbol], symbol
                    )

        # Atualiza o cache com os dados tratados
        self._latest_symbol_data = latest_data

        # Avança para o próximo timestamp
        self._timestamp_index += 1

        # Verifica se ainda há timestamps após este
        if self._timestamp_index >= len(self._current_timestamps):
            if self._continue_backtest:  # Imprime apenas uma vez
                logger.info("Este foi o último timestamp.")
            self._continue_backtest = False

        # Cria e retorna o MarketEvent, mesmo que _latest_symbol_data esteja vazio para este ts
        return MarketEvent(timestamp=current_timestamp)

    def get_all_bars_between(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        symbol: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Retorna todas as barras disponíveis entre duas datas para um ou todos os símbolos.

        Esta função busca dados diretamente do banco de dados SQL para o intervalo de datas
        especificado, permitindo filtrar por um símbolo específico ou retornar dados para
        todos os símbolos gerenciados por este handler.

        Args:
            start_date (datetime.datetime): A data de início do período (inclusive).
            end_date (datetime.datetime): A data de término do período (inclusive).
            symbol (Optional[str]): Símbolo específico para filtrar. Se None, retorna
                                    dados para todos os símbolos gerenciados.

        Returns:
            pl.DataFrame: DataFrame Polars contendo todas as barras dentro do intervalo
                          com colunas padronizadas para o formato interno.
                          Retorna um DataFrame vazio se não houver dados no intervalo.
        """
        if not self._db_connection:
            logger.error(
                f"Erro: Sem conexão com o banco para buscar barras entre {start_date} e {end_date}."
            )
            return pl.DataFrame()  # Retorna DataFrame vazio

        logger.info(
            f"Buscando todas as barras entre {start_date} e {end_date}"
            + (f" para {symbol}" if symbol else " para todos os símbolos")
        )

        # Colunas a selecionar do banco
        db_cols_to_select = [self._symbol_col, self._time_col] + self._ohlcv_cols
        select_clause = ", ".join([f'"{col}"' for col in db_cols_to_select])

        # Filtro de símbolo
        symbol_filter = ""
        params = [start_date, end_date]
        if symbol:
            symbol_filter = f'AND "{self._symbol_col}" = ?'
            params.append(symbol)
        else:
            # Filtro para a lista de símbolos gerenciados por este handler
            placeholders = ", ".join(["?" for _ in self._symbol_list])
            symbol_filter = f'AND "{self._symbol_col}" IN ({placeholders})'
            params.extend(self._symbol_list)

        try:
            query = f"""
            SELECT {select_clause}
            FROM "{self._table_name}"
            WHERE "{self._time_col}" >= ? AND "{self._time_col}" <= ?
              {symbol_filter}
            ORDER BY "{self._symbol_col}", "{self._time_col}" ASC;
            """
            logger.debug(f"Executando query entre datas: {query}")
            logger.debug(f"Parâmetros: {params}")

            # Aqui usamos uma função auxiliar que deve ser implementada para executar
            # a query e retornar um DataFrame Polars
            try:
                df_all_bars = self._execute_query_to_polars(query, params)
            except AttributeError:
                # Caso o método auxiliar não esteja implementado, criamos uma implementação básica
                # Este é um exemplo de como implementar _execute_query_to_polars
                logger.debug(
                    "Método _execute_query_to_polars não encontrado. Implementando temporariamente."
                )

                # Implementação genérica que deve ser adaptada ao driver SQL específico

                # Supondo que self._db_connection seja um objeto de conexão compatível
                cursor = self._db_connection.cursor()
                cursor.execute(query, params)

                # Obter todos os registros
                rows = cursor.fetchall()

                # Obter nomes das colunas
                column_names = [column[0] for column in cursor.description]

                # Fechar o cursor
                cursor.close()

                # Criar DataFrame Polars
                if rows:
                    # Converter para um formato que o Polars aceite
                    data_dict = {
                        column_names[i]: [row[i] for row in rows]
                        for i in range(len(column_names))
                    }
                    df_all_bars = pl.DataFrame(data_dict)
                else:
                    # DataFrame vazio com schema correto
                    df_all_bars = pl.DataFrame(
                        schema={
                            col: pl.Float64 for col in self._ohlcv_cols_map.values()
                        }
                    )
                    df_all_bars = df_all_bars.with_columns(
                        [
                            pl.lit("").alias(self._symbol_col),
                            pl.lit(datetime.datetime.now()).alias(self._time_col),
                        ]
                    )
                    # Remover a única linha que foi criada
                    df_all_bars = df_all_bars.filter(pl.lit(False))

            # Renomear colunas para o padrão interno
            rename_map = {**self._ohlcv_cols_map, self._time_col: "date"}
            # Apenas renomear colunas que existem no DataFrame
            rename_map = {
                k: v for k, v in rename_map.items() if k in df_all_bars.columns
            }

            if rename_map:
                df_renamed = df_all_bars.rename(rename_map)
            else:
                df_renamed = df_all_bars

            # Certifique-se de que as colunas tenham os tipos corretos
            # Isto pode ser necessário dependendo do driver SQL e como ele mapeia tipos
            df_typed = df_renamed
            if "date" in df_renamed.columns and not pl.datatypes.is_temporal(
                df_renamed["date"].dtype
            ):
                df_typed = df_renamed.with_columns(
                    [pl.col("date").cast(pl.Datetime, strict=False)]
                )

            for col in ["open", "high", "low", "close", "volume"]:
                if col in df_typed.columns and df_typed[col].dtype != pl.Float64:
                    df_typed = df_typed.with_columns(
                        [pl.col(col).cast(pl.Float64, strict=False)]
                    )

            # Aplicar tratamento para valores ausentes e validação OHLC
            if symbol:
                # Se estamos buscando dados para um símbolo específico
                df_processed = self._fill_missing_values(
                    df_typed, symbol, method=self._fill_missing_method
                )
                if self._validate_ohlc:
                    df_processed = self._validate_ohlc_integrity(df_processed, symbol)
                # Tratar outliers conforme configuração
                if self._outlier_detection.get("method") is not None:
                    df_processed = self._handle_outliers(df_processed, symbol)
            else:
                # Se estamos buscando dados para múltiplos símbolos
                # Processar cada símbolo separadamente e depois recombinar
                symbol_dfs = []
                for sym in self._symbol_list:
                    # Filtrar apenas os dados deste símbolo
                    sym_df = df_typed.filter(pl.col("symbol") == sym)
                    if not sym_df.is_empty():
                        # Aplicar tratamento e validação
                        sym_df = self._fill_missing_values(
                            sym_df, sym, method=self._fill_missing_method
                        )
                        if self._validate_ohlc:
                            sym_df = self._validate_ohlc_integrity(sym_df, sym)
                        # Tratar outliers conforme configuração
                        if self._outlier_detection.get("method") is not None:
                            sym_df = self._handle_outliers(sym_df, sym)
                        symbol_dfs.append(sym_df)

                # Recombinar os DataFrames processados
                if symbol_dfs:
                    df_processed = pl.concat(symbol_dfs)
                else:
                    df_processed = df_typed

            logger.info(
                f"Dados recuperados entre {start_date} e {end_date}: {df_processed.shape}"
            )
            return df_processed

        except Exception as e:
            logger.error(f"Erro ao buscar barras entre {start_date} e {end_date}: {e}")
            return pl.DataFrame()  # Retorna DataFrame vazio em caso de erro

    def get_all_bars_in_period(self, symbol: str, period: str) -> pl.DataFrame:
        """
        Retorna todas as barras disponíveis para um símbolo em um período específico.

        Esta função converte uma string de período (ex: '1y', '6mo', '30d') em um
        intervalo de datas e retorna todas as barras dentro desse intervalo para o
        símbolo especificado, usando get_all_bars_between internamente.

        Args:
            symbol (str): O símbolo do ativo para o qual buscar os dados.
            period (str): String que representa o período no formato:
                          - 'Ny' para N anos (ex: '1y')
                          - 'Nmo' para N meses (ex: '6mo')
                          - 'Nd' para N dias (ex: '30d')
                          - 'Nw' para N semanas (ex: '2w')

        Returns:
            pl.DataFrame: DataFrame Polars contendo as barras dentro do período especificado
                          para o símbolo solicitado. Retorna um DataFrame vazio se o símbolo
                          não for encontrado ou se não houver dados no período.

        Raises:
            ValueError: Se o formato do período for inválido ou não suportado.
        """
        logger.info(f"Buscando barras para {symbol} no período {period}")

        # Usar a data final configurada no handler (ou data atual se não especificada)
        end_date = self._end_date if self._end_date else datetime.datetime.now()

        # Parsear a string de período para calcular a data de início
        try:
            # Extrair o número e a unidade do período
            import re

            match = re.match(r"(\d+)([a-zA-Z]+)", period)
            if not match:
                raise ValueError(f"Formato de período inválido: {period}")

            value = int(match.group(1))
            unit = match.group(2).lower()

            # Calcular a data de início baseada na unidade
            from dateutil.relativedelta import relativedelta

            if unit == "y":  # Anos
                start_date = end_date - relativedelta(years=value)
            elif unit == "mo":  # Meses
                start_date = end_date - relativedelta(months=value)
            elif unit == "w":  # Semanas
                start_date = end_date - datetime.timedelta(weeks=value)
            elif unit == "d":  # Dias
                start_date = end_date - datetime.timedelta(days=value)
            else:
                raise ValueError(f"Unidade de período não suportada: {unit}")

            logger.info(f"Período {period} calculado como: {start_date} até {end_date}")

            # Usar get_all_bars_between para obter os dados
            return self.get_all_bars_between(
                start_date=start_date, end_date=end_date, symbol=symbol
            )

        except Exception as e:
            logger.error(f"Erro ao processar período '{period}' para {symbol}: {e}")
            raise ValueError(f"Erro ao processar período: {e}")

    def close_connection(self):
        """Fecha a conexão com o banco de dados, se aplicável."""
        if hasattr(self, "_db_connection") and self._db_connection:
            try:
                # A forma de fechar depende do objeto de conexão (ex: SQLAlchemy connection)
                logger.info("Tentando fechar conexão com o banco de dados...")
                # Exemplo: self._db_connection.close()
                # Certifique-se de que o método close() exista e seja chamado corretamente
                if hasattr(self._db_connection, "close") and callable(
                    self._db_connection.close
                ):
                    self._db_connection.close()
                    logger.info("Conexão com o banco de dados fechada.")
                    self._db_connection = None  # Define como None após fechar
                else:
                    logger.warning(
                        "Aviso: Objeto de conexão não possui um método close() esperado."
                    )
            except Exception as e:
                logger.error(f"Erro ao fechar conexão com o banco de dados: {e}")

    def __del__(self):
        """Garante que a conexão seja fechada quando o objeto for destruído."""
        # print(f"Destruindo SQLDataHandler...") # Para debug
        self.close_connection()

    def _validate_ohlc_integrity(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Valida a integridade dos dados OHLC e corrige inconsistências quando possível.

        Verificações realizadas:
        1. High >= Low (obrigatório)
        2. Open entre High e Low (ou igual a um deles)
        3. Close entre High e Low (ou igual a um deles)
        4. Volume não negativo

        Args:
            df (pl.DataFrame): DataFrame Polars contendo dados OHLC
            symbol (str): Símbolo dos dados sendo validados (para logs)

        Returns:
            pl.DataFrame: DataFrame com dados corrigidos quando possível
        """
        if df.is_empty():
            return df

        original_count = df.height
        issues_count = 0

        # Verificar se temos colunas OHLC antes de validar
        cols_to_check = ["open", "high", "low", "close", "volume"]
        cols_present = [c for c in cols_to_check if c in df.columns]

        # Se não temos as colunas necessárias, retornar sem validar
        if not all(c in cols_present for c in ["high", "low"]):
            logger.warning(
                f"Não é possível validar OHLC para {symbol}, colunas high/low ausentes"
            )
            return df

        # Verificar se high >= low
        high_low_invalid = df.filter(pl.col("high") < pl.col("low"))
        if not high_low_invalid.is_empty():
            issues_count += high_low_invalid.height
            logger.warning(
                f"{high_low_invalid.height} barras com High < Low para {symbol}. Corrigindo..."
            )

            # Corrigir invertendo high e low
            df = (
                df.with_columns(
                    [
                        pl.when(pl.col("high") < pl.col("low"))
                        .then(pl.struct(high=pl.col("low"), low=pl.col("high")))
                        .otherwise(pl.struct(high=pl.col("high"), low=pl.col("low")))
                        .alias("temp")
                    ]
                )
                .with_columns(
                    [
                        pl.col("temp").struct.field("high").alias("high"),
                        pl.col("temp").struct.field("low").alias("low"),
                    ]
                )
                .drop("temp")
            )

        # Verificar se open está entre high e low
        if "open" in cols_present:
            open_invalid = df.filter(
                (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low"))
            )
            if not open_invalid.is_empty():
                issues_count += open_invalid.height
                logger.warning(
                    f"{open_invalid.height} barras com Open fora do intervalo High-Low para {symbol}. Corrigindo..."
                )

                # Corrigir limitando open entre high e low
                df = df.with_columns(
                    [
                        pl.when(pl.col("open") > pl.col("high"))
                        .then(pl.col("high"))
                        .when(pl.col("open") < pl.col("low"))
                        .then(pl.col("low"))
                        .otherwise(pl.col("open"))
                        .alias("open")
                    ]
                )

        # Verificar se close está entre high e low
        if "close" in cols_present:
            close_invalid = df.filter(
                (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))
            )
            if not close_invalid.is_empty():
                issues_count += close_invalid.height
                logger.warning(
                    f"{close_invalid.height} barras com Close fora do intervalo High-Low para {symbol}. Corrigindo..."
                )

                # Corrigir limitando close entre high e low
                df = df.with_columns(
                    [
                        pl.when(pl.col("close") > pl.col("high"))
                        .then(pl.col("high"))
                        .when(pl.col("close") < pl.col("low"))
                        .then(pl.col("low"))
                        .otherwise(pl.col("close"))
                        .alias("close")
                    ]
                )

        # Verificar volume negativo
        if "volume" in cols_present:
            volume_invalid = df.filter(pl.col("volume") < 0)
            if not volume_invalid.is_empty():
                issues_count += volume_invalid.height
                logger.warning(
                    f"{volume_invalid.height} barras com Volume negativo para {symbol}. Corrigindo..."
                )

                # Corrigir definindo volumes negativos como 0
                df = df.with_columns(
                    [
                        pl.when(pl.col("volume") < 0)
                        .then(0.0)
                        .otherwise(pl.col("volume"))
                        .alias("volume")
                    ]
                )

        if issues_count > 0:
            logger.info(
                f"Total de {issues_count} problemas de integridade OHLCV corrigidos para {symbol}"
            )

        # Remover duplicatas com base na coluna de tempo, se existir
        time_cols = [c for c in ["date", "datetime", "timestamp"] if c in df.columns]
        if time_cols:
            before = df.height
            df = df.unique(subset=time_cols)
            after = df.height
            if after < before:
                logger.info(
                    f"Removidas {before - after} duplicatas em {symbol} com base em {time_cols}"
                )

        # Checar timezone awareness
        for col in time_cols:
            if pl.datatypes.is_temporal(df[col].dtype):
                if (
                    hasattr(df[col], "dtype")
                    and hasattr(df[col].dtype, "tz")
                    and df[col].dtype.tz is not None
                ):
                    logger.debug(
                        f"Coluna {col} de {symbol} já é timezone-aware: {df[col].dtype.tz}"
                    )
                else:
                    logger.debug(
                        f"Coluna {col} de {symbol} não é timezone-aware. Considere converter para UTC."
                    )

        return df

    def _fill_missing_values(
        self, df: pl.DataFrame, symbol: str, method: str = "ffill_bfill"
    ) -> pl.DataFrame:
        """
        Preenche valores ausentes nas colunas OHLCV usando várias estratégias.

        Args:
            df (pl.DataFrame): DataFrame Polars contendo dados OHLCV
            symbol (str): Símbolo dos dados sendo tratados (para logs)
            method (str): Método de preenchimento:
                - 'ffill_bfill': Forward fill seguido de backward fill (padrão)
                - 'interpolate': Interpolação linear entre pontos
                - 'prev_day': Usar valores do dia anterior (útil para dados diários)
                - 'zeros': Preencher com zeros (mais adequado para volume)
                - 'drop': Remover linhas com valores ausentes

        Returns:
            pl.DataFrame: DataFrame com valores ausentes preenchidos
        """
        if df.is_empty():
            return df

        original_count = df.height
        null_counts = {}

        # Colunas a verificar (usar nomes padrão internos)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        ohlcv_cols_present = [col for col in ohlcv_cols if col in df.columns]

        if not ohlcv_cols_present:
            return df  # Sem colunas OHLCV, retorna DataFrame original

        # Contar valores nulos em cada coluna
        for col in ohlcv_cols_present:
            null_count = df[col].is_null().sum()
            if null_count > 0:
                null_counts[col] = null_count

        if not null_counts:
            return df  # Sem valores nulos

        logger.info(f"Valores ausentes detectados em {symbol}: {null_counts}")

        # Aplicar estratégia de preenchimento
        if method == "drop":
            # Método simples: remover linhas com valores nulos
            df = df.drop_nulls(subset=ohlcv_cols_present)
            logger.info(
                f"Removidas {original_count - df.height} linhas com valores ausentes para {symbol}"
            )

        elif method == "zeros":
            # Preencher com zeros (útil para volume)
            for col in ohlcv_cols_present:
                if col in null_counts:
                    df = df.with_columns([pl.col(col).fill_null(0.0).alias(col)])
            logger.info(
                f"Preenchidos {sum(null_counts.values())} valores ausentes com zeros para {symbol}"
            )

        elif method == "interpolate":
            # Interpolação linear (requer dados ordenados cronologicamente)
            for col in ohlcv_cols_present:
                if col in null_counts:
                    # Polars implementou interpolate na versão mais recente
                    try:
                        df = df.with_columns([pl.col(col).interpolate().alias(col)])
                    except AttributeError:
                        # Fallback para versões anteriores do Polars
                        df = df.with_columns(
                            [pl.col(col).fill_null(strategy="linear").alias(col)]
                        )
            logger.info(
                f"Interpolados {sum(null_counts.values())} valores ausentes para {symbol}"
            )

        elif method == "prev_day":
            # Para dados diários: preencher com valor do dia anterior
            # Primeiro forward fill, depois backward fill para os primeiros valores
            for col in ohlcv_cols_present:
                if col in null_counts:
                    df = df.with_columns(
                        [pl.col(col).fill_null(strategy="forward").alias(col)]
                    )
                    # Preencher começo da série se necessário
                    df = df.with_columns(
                        [pl.col(col).fill_null(strategy="backward").alias(col)]
                    )
            logger.info(
                f"Preenchidos {sum(null_counts.values())} valores ausentes com método prev_day para {symbol}"
            )

        else:  # método padrão: ffill_bfill
            # Forward fill seguido de backward fill
            for col in ohlcv_cols_present:
                if col in null_counts:
                    df = df.with_columns(
                        [pl.col(col).fill_null(strategy="forward").alias(col)]
                    )
                    # Preencher valores no início que não puderam usar forward fill
                    df = df.with_columns(
                        [pl.col(col).fill_null(strategy="backward").alias(col)]
                    )
            logger.info(
                f"Preenchidos {sum(null_counts.values())} valores ausentes com ffill_bfill para {symbol}"
            )

        # Verificar se ainda existem nulos após o processamento
        remaining_nulls = 0
        for col in ohlcv_cols_present:
            remaining_nulls += df[col].is_null().sum()

        if remaining_nulls > 0:
            logger.warning(
                f"Atenção: Ainda existem {remaining_nulls} valores ausentes em {symbol} após o preenchimento"
            )
            # Como último recurso, remover linhas com valores nulos remanescentes
            df = df.drop_nulls(subset=ohlcv_cols_present)
            logger.info(
                f"Removidas {original_count - df.height} linhas com valores ausentes persistentes"
            )

        return df

    def _handle_outliers(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Detecta e trata outliers nas colunas OHLCV.

        Args:
            df (pl.DataFrame): DataFrame Polars contendo dados OHLCV
            symbol (str): Símbolo dos dados sendo tratados (para logs)

        Returns:
            pl.DataFrame: DataFrame com outliers tratados, conforme configuração
        """
        if df.is_empty():
            return df

        method = self._outlier_detection.get("method")
        if method is None:
            return df  # Detecção de outliers desativada

        threshold = self._outlier_detection.get("threshold", 3.0)
        action = self._outlier_detection.get("action", "clip")
        columns = self._outlier_detection.get(
            "columns", ["open", "high", "low", "close", "volume"]
        )

        # Filtra apenas colunas presentes no DataFrame
        columns = [col for col in columns if col in df.columns]
        if not columns:
            return df  # Nenhuma coluna para tratar

        outliers_found = 0
        result_df = df.clone()

        for col in columns:
            if method == "std_dev":
                # Método de desvio padrão
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

            elif method == "iqr":
                # Método de intervalo interquartil
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

            elif method == "absolute":
                # Método de limites absolutos (threshold interpretado como percentual)
                # Útil quando se conhece os limites aceitáveis do mercado
                median = df[col].median()
                lower_bound = median * (1 - threshold / 100)
                upper_bound = median * (1 + threshold / 100)

                # Garantir que volumes e preços não fiquem negativos
                if col in ["volume", "open", "high", "low", "close"]:
                    lower_bound = max(0, lower_bound)
            else:
                # Método não reconhecido, pular coluna
                continue

            # Detectar outliers (valores fora dos limites)
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_count = outliers_mask.sum()

            if outliers_count > 0:
                outliers_found += outliers_count

                if action == "clip":
                    # Limitar valores aos limites superior e inferior
                    result_df = result_df.with_columns(
                        [
                            pl.when(pl.col(col) < lower_bound)
                            .then(lower_bound)
                            .when(pl.col(col) > upper_bound)
                            .then(upper_bound)
                            .otherwise(pl.col(col))
                            .alias(col)
                        ]
                    )

                elif action == "remove":
                    # Remover linhas com outliers
                    result_df = result_df.filter(~outliers_mask)

                # action == 'ignore' não faz nada, apenas reporta

        if outliers_found > 0:
            msg = f"Detectados {outliers_found} outliers em {symbol} usando método '{method}'"
            if action == "clip":
                msg += " (valores limitados aos limites)"
            elif action == "remove":
                msg += f" ({df.height - result_df.height} linhas removidas)"
            else:
                msg += " (apenas detectados, não tratados)"
            logger.info(msg)

        return result_df
