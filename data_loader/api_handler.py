import datetime as dt
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import polars as pl
import requests

from events import MarketEvent

from .base import DataHandler

logger = logging.getLogger("PolygonDataHandler")

ASSET_PREFIX = {
    "stocks": "",  # ex: AAPL
    "crypto": "X:",  # ex: X:BTCUSD
    "forex": "C:",  # ex: C:EURUSD
    "indices": "I:",  # ex: I:SPX
}

MAX_WINDOW_DAYS = 185  # ~6 meses
MAX_ATTEMPTS = 5


class PolygonDataHandler(DataHandler):
    """
    Manipulador incremental de dados da API Polygon.io, compatível com o fluxo de backtesting.
    Suporta candles de qualquer intervalo e múltiplos tipos de ativos (ações, cripto, forex, índices).
    """

    BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    TICKER_VALIDATE_URL = "https://api.polygon.io/v3/reference/tickers/{symbol}"
    _cache_lock = threading.Lock()

    def __init__(
        self,
        api_key: str,
        symbol_list: List[str],
        asset_types: Dict[str, str],
        multiplier: int = 1,
        timespan: str = "minute",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cache_dir: Optional[str] = None,
        save_csv: bool = False,
        max_workers: int = 4,
        request_delay_seconds: float = 12.0,
    ):
        self._api_key = api_key
        self._symbol_list = symbol_list
        self.asset_types = asset_types
        self._multiplier = multiplier
        self._timespan = timespan.strip().lower()
        self._start_date = start_date
        self._end_date = end_date
        self._cache_dir = cache_dir
        self._save_csv = save_csv
        self._max_workers = max_workers
        self._request_delay_seconds = request_delay_seconds
        self._data: Dict[str, pl.DataFrame] = {}
        self._global_time_index: Optional[pl.Series] = None
        self._current_time_idx_ptr: int = 0
        self._latest_symbol_bars: Dict[str, Optional[pl.DataFrame]] = {
            s: None for s in symbol_list
        }
        self._current_datetime: Optional[datetime] = None
        self.continue_backtest_flag: bool = False
        self._load_or_download_all()

    @property
    def symbols(self) -> List[str]:
        return self._symbol_list

    @property
    def continue_backtest(self) -> bool:
        return self.continue_backtest_flag

    def _cache_paths(self, symbol: str):
        if not self._cache_dir:
            return None, None
        os.makedirs(self._cache_dir, exist_ok=True)
        fname = f"{symbol}_{self._multiplier}{self._timespan}.parquet"
        parquet_path = os.path.join(self._cache_dir, fname)
        csv_path = parquet_path.replace(".parquet", ".csv")
        return parquet_path, csv_path

    def _make_url(
        self, asset_type: str, symbol: str, start_date: str, end_date: str
    ) -> str:
        prefix = ASSET_PREFIX.get(asset_type, "")
        full_symbol = f"{prefix}{symbol}"
        return self.BASE_URL.format(
            symbol=full_symbol,
            multiplier=self._multiplier,
            timespan=self._timespan,
            start_date=start_date,
            end_date=end_date,
        )

    def _validate_symbol(self, symbol: str, asset_type: str) -> bool:
        prefix = ASSET_PREFIX.get(asset_type, "")
        full_symbol = f"{prefix}{symbol}"
        url = self.TICKER_VALIDATE_URL.format(symbol=full_symbol)
        params = {"apiKey": self._api_key, "date": self._start_date}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return True
            logger.warning(
                f"Ticker inválido ou não encontrado: {full_symbol} ({resp.status_code})"
            )
            return False
        except Exception as e:
            logger.warning(f"Erro ao validar ticker {full_symbol}: {e}")
            return False

    def _date_range_windows(
        self, start: str, end: str, window_days: int = MAX_WINDOW_DAYS
    ):
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        curr = start_dt
        while curr < end_dt:
            next_dt = min(curr + timedelta(days=window_days), end_dt)
            yield curr.strftime("%Y-%m-%d"), next_dt.strftime("%Y-%m-%d")
            curr = next_dt + timedelta(days=1)  # não repetir o dia final

    def _load_or_download_all(self):
        logger.info(
            f"Iniciando carregamento dos dados para {len(self._symbol_list)} ativos..."
        )
        results = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._load_or_download_symbol, symbol): symbol
                for symbol in self._symbol_list
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    results[symbol] = df
                    logger.info(
                        f"Dados carregados para {symbol}: {df.height if hasattr(df, 'height') else len(df)} linhas."
                    )
                except Exception as e:
                    logger.error(f"Erro ao carregar dados para {symbol}: {e}")
                    results[symbol] = pl.DataFrame([])
        self._data = results

        # Modificação para tratar lista vazia antes de pl.concat
        date_series_list = [
            df["date"]
            for df in self._data.values()
            if isinstance(df, pl.DataFrame) and df.height > 0 and "date" in df.columns
        ]

        if date_series_list:
            all_times = pl.concat(date_series_list).unique().sort()
        else:
            # Usar dtype=pl.Datetime para consistência, mesmo que vazia
            all_times = pl.Series([], dtype=pl.Datetime)

        self._global_time_index = all_times
        self._current_time_idx_ptr = 0
        self.continue_backtest_flag = (
            self._global_time_index.len()
            if hasattr(self._global_time_index, "len")
            else len(self._global_time_index)
        ) > 0

    def _download_data_in_range(
        self,
        symbol: str,
        asset_type: str,
        range_start_date_str: str,
        range_end_date_str: str,
    ) -> pl.DataFrame:
        """Baixa dados para um símbolo dentro de um intervalo de datas específico."""
        logger.info(
            f"Iniciando download em _download_data_in_range para {symbol} de {range_start_date_str} a {range_end_date_str}"
        )
        if not range_start_date_str or not range_end_date_str:
            logger.warning(
                f"Datas de início ou fim não fornecidas para {symbol} em _download_data_in_range."
            )
            return pl.DataFrame([])

        # Validação para evitar que start_date seja maior que end_date
        try:
            start_dt = datetime.strptime(range_start_date_str, "%Y-%m-%d")
            end_dt = datetime.strptime(range_end_date_str, "%Y-%m-%d")
            if start_dt > end_dt:
                logger.warning(
                    f"Data de início ({range_start_date_str}) é posterior à data de fim ({range_end_date_str}) para {symbol}."
                )
                return pl.DataFrame([])
        except ValueError as e:
            logger.error(
                f"Formato de data inválido em _download_data_in_range para {symbol}: {e}"
            )
            return pl.DataFrame([])

        dfs_period = []
        for win_start, win_end in self._date_range_windows(
            range_start_date_str, range_end_date_str
        ):
            logger.info(
                f"Download da janela dentro de _download_data_in_range: {win_start} até {win_end} para {symbol}"
            )
            df_win = self._download_symbol(
                symbol, asset_type, win_start, win_end
            )  # Pass asset_type
            if hasattr(df_win, "height") and df_win.height > 0:
                dfs_period.append(df_win)

        if not dfs_period:
            logger.info(
                f"Nenhum dado baixado no período {range_start_date_str}-{range_end_date_str} para {symbol}."
            )
            return pl.DataFrame([])

        df_period_full = pl.concat(dfs_period)
        if "date" in df_period_full.columns and df_period_full.height > 0:
            df_period_full = df_period_full.unique(subset=["date"], keep="first").sort(
                "date"
            )
        logger.info(
            f"Download concluído em _download_data_in_range para {symbol}: {df_period_full.height} linhas de {range_start_date_str} a {range_end_date_str}"
        )
        return df_period_full

    def _load_or_download_symbol(self, symbol: str) -> pl.DataFrame:
        asset_type = self.asset_types.get(symbol, "stocks")
        logger.info(f"Validando ticker: {symbol} ({asset_type})")
        if not self._validate_symbol(symbol, asset_type):
            logger.error(
                f"Ticker inválido: {symbol} ({asset_type}) - pulando download."
            )
            return pl.DataFrame([])

        parquet_path, csv_path = self._cache_paths(symbol)
        cached_df: Optional[pl.DataFrame] = None
        df_prefix: Optional[pl.DataFrame] = None
        df_suffix: Optional[pl.DataFrame] = None

        # Garantir que _start_date e _end_date (da instância) sejam válidos
        if not self._start_date or not self._end_date:
            logger.error(
                f"Datas de início ou fim da instância não definidas para {symbol}."
            )
            return pl.DataFrame([])
        try:
            # Converter para datetime e tornar timezone-aware (UTC)
            datetime_self_start_date = dt.datetime.strptime(
                self._start_date, "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            datetime_self_end_date = dt.datetime.strptime(
                self._end_date, "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
        except ValueError as e:
            logger.error(
                f"Formato de data inválido nas datas da instância para {symbol}: {e}"
            )
            return pl.DataFrame([])

        if parquet_path and os.path.exists(parquet_path):
            try:
                logger.info(
                    f"Tentando ler cache Parquet para {symbol} em {parquet_path}"
                )
                cached_df = pl.read_parquet(parquet_path)
                logger.info(
                    f"Cache carregado de {parquet_path} para {symbol}: {cached_df.height} linhas"
                )

                # 1. Verificar se as datas da instância estão completamente contidas no cache
                # cache_min_date = cached_df["date"].min() # Removido
                # cache_max_date = cached_df["date"].max() # Removido

                # Conversão para objetos datetime cientes do fuso horário (UTC)
                if cached_df["date"].dtype == pl.String:  # Correção aqui
                    logger.info(
                        f"Coluna 'date' no cache de {symbol} é String. Convertendo para Datetime."
                    )
                    cached_df = cached_df.with_columns(
                        pl.col("date").str.to_datetime(time_unit="ms")
                    )

                # Garantir que a coluna 'date' do cache seja UTC
                if (
                    cached_df["date"].dtype == pl.Datetime
                    and cached_df["date"].dt.time_zone() is None
                ):
                    logger.info(
                        f"Coluna 'date' no cache de {symbol} é naive. Localizando para UTC."
                    )
                    cached_df = cached_df.with_columns(
                        pl.col("date").dt.replace_time_zone("UTC")
                    )
                elif cached_df["date"].dtype == pl.Datetime:
                    logger.debug(
                        f"Coluna 'date' no cache de {symbol} já possui timezone: {cached_df['date'].dt.time_zone()}"
                    )

            except Exception as e:
                logger.warning(
                    f"Falha ao ler ou processar cache Parquet para {symbol}: {e}. Tratando como se não houvesse cache."
                )
                cached_df = None

        dfs_to_concat = []

        if cached_df is not None and cached_df.height > 0:
            logger.info(f"Cache existente para {symbol} com {cached_df.height} linhas.")
            logger.info(
                f"Período solicitado: {datetime_self_start_date} a {datetime_self_end_date}"
            )

            # 1. Baixar dados ANTES do cache, se necessário
            if datetime_self_start_date < cached_df["date"].min():
                prefix_end_date_obj = cached_df["date"].min() - timedelta(days=1)
                prefix_end_date_str = prefix_end_date_obj.strftime("%Y-%m-%d")
                logger.info(
                    f"Necessário baixar dados de prefixo para {symbol} de {self._start_date} a {prefix_end_date_str}"
                )
                # Evitar baixar se o range for inválido
                if (
                    datetime.strptime(self._start_date, "%Y-%m-%d")
                    <= prefix_end_date_obj
                ):
                    df_prefix = self._download_data_in_range(
                        symbol, asset_type, self._start_date, prefix_end_date_str
                    )
                    if df_prefix is not None and df_prefix.height > 0:
                        dfs_to_concat.append(df_prefix)
                else:
                    logger.info(
                        f"Range de prefixo inválido para {symbol}: {self._start_date} > {prefix_end_date_str}. Pulando download de prefixo."
                    )

            dfs_to_concat.append(cached_df)  # Adicionar o cache existente

            # 2. Baixar dados DEPOIS do cache, se necessário
            if datetime_self_end_date > cached_df["date"].max():
                suffix_start_date_obj = cached_df["date"].max() + timedelta(days=1)
                suffix_start_date_str = suffix_start_date_obj.strftime("%Y-%m-%d")
                logger.info(
                    f"Necessário baixar dados de sufixo para {symbol} de {suffix_start_date_str} a {self._end_date}"
                )
                # Evitar baixar se o range for inválido
                if suffix_start_date_obj <= datetime.strptime(
                    self._end_date, "%Y-%m-%d"
                ):
                    df_suffix = self._download_data_in_range(
                        symbol, asset_type, suffix_start_date_str, self._end_date
                    )
                    if df_suffix is not None and df_suffix.height > 0:
                        dfs_to_concat.append(df_suffix)
                else:
                    logger.info(
                        f"Range de sufixo inválido para {symbol}: {suffix_start_date_str} > {self._end_date}. Pulando download de sufixo."
                    )

        else:  # Nenhum cache válido encontrado, baixar tudo
            logger.info(
                f"Nenhum cache válido para {symbol}. Baixando dados completos de {self._start_date} a {self._end_date}"
            )
            df_full_download = self._download_data_in_range(
                symbol, asset_type, self._start_date, self._end_date
            )
            if df_full_download is not None and df_full_download.height > 0:
                dfs_to_concat.append(df_full_download)

        if not dfs_to_concat:
            logger.warning(
                f"Nenhum dado para concatenar para o símbolo {symbol} após tentativa de download e leitura de cache."
            )
            return pl.DataFrame([])

        # Concatenar todos os dataframes coletados
        df_final = pl.concat(dfs_to_concat)

        if df_final.height == 0:
            logger.warning(
                f"DataFrame final para {symbol} está vazio após concatenação."
            )
            return pl.DataFrame([])

        # Garantir que a coluna 'date' seja datetime antes de unique/sort
        # Se veio de _download_data_in_range, já deve ser datetime
        # Se veio do cache e era string/int, a tentativa de conversão ocorreu acima.
        # Adicionar uma verificação final ou garantir que _download_symbol sempre retorne datetime.
        # A função _download_symbol já converte 't' para datetime, então está ok.
        # E _download_data_in_range já garante que a coluna 'date' é datetime.

        df_final = df_final.unique(subset=["date"], keep="first").sort("date")
        logger.info(f"Processamento final para {symbol}: {df_final.height} linhas.")

        if parquet_path:
            try:
                with self._cache_lock:  # Usar o lock para escrita
                    logger.info(
                        f"Salvando dados combinados em Parquet para {symbol} em {parquet_path}"
                    )
                    df_final.write_parquet(parquet_path, use_pyarrow=True)
                    logger.info(f"Salvo com sucesso em Parquet para {symbol}.")
            except Exception as e:
                logger.error(f"Falha ao salvar Parquet para {symbol}: {e}")

        if self._save_csv and csv_path:
            try:
                with self._cache_lock:  # Usar o lock para escrita
                    logger.info(
                        f"Salvando dados combinados em CSV para {symbol} em {csv_path}"
                    )
                    df_final.write_csv(csv_path)
                    logger.info(f"Salvo com sucesso em CSV para {symbol}.")
            except Exception as e:
                logger.warning(f"Falha ao salvar CSV para {symbol}: {e}")

        return df_final

    # Adicionado asset_type
    def _download_symbol(
        self,
        symbol: str,
        asset_type: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pl.DataFrame:
        # asset_type = self.asset_types.get(symbol, "stocks") # Removido, pois é passado como argumento
        url = self._make_url(asset_type, symbol, start_date, end_date)
        base_params = {
            "apiKey": self._api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        params = base_params.copy()
        all_data = []
        attempt = 0
        # Número de tentativas se 'results' estiver vazio, mas 'next_url' existir
        retries_on_empty = 2
        current_empty_retry = 0

        while url:
            try:
                logger.info(f"Requesting URL: {url} for symbol {symbol}")
                # logger.info(f"Request params: {params}") # Pode ser muito verboso
                resp = requests.get(url, params=params, timeout=30)

                if (
                    resp.status_code == 429 or resp.status_code >= 500
                ):  # Rate limit ou erro de servidor
                    if attempt < MAX_ATTEMPTS:
                        # Adiciona jitter
                        wait = (2**attempt) + (random.uniform(0, 1))
                        logger.warning(
                            f"Rate limit/Server error ({resp.status_code}) para {symbol}. Tentativa {attempt + 1}/{MAX_ATTEMPTS}. Esperando {wait:.2f}s..."
                        )
                        time.sleep(wait)
                        attempt += 1
                        # Não limpar params aqui, a next_url já deve ter os params corretos ou params será recriado
                        continue
                    else:
                        logger.error(
                            f"Falha definitiva ({resp.status_code}) após {MAX_ATTEMPTS} tentativas para {symbol} na URL: {url}."
                        )
                        # Retorna DF vazio em falha definitiva
                        return pl.DataFrame([])

                if resp.status_code != 200:
                    logger.error(
                        f"Erro ao requisitar dados para {symbol}: {resp.status_code} {resp.text} na URL: {url}"
                    )
                    # Se for um erro de autenticação ou 'ticker not found', pode não adiantar tentar de novo com next_url
                    if (
                        resp.status_code == 401
                        or resp.status_code == 403
                        or "ticker not found" in resp.text.lower()
                    ):
                        return pl.DataFrame([])
                    # Para outros erros 4xx, vamos tentar continuar se houver next_url, mas logar como erro.
                    # Se não houver next_url, quebra o loop abaixo.

                data = resp.json()

                results = data.get("results")
                next_page_url = data.get("next_url")

                if not results and next_page_url:
                    # Às vezes a API retorna 'results' vazio mas com 'next_url'.
                    # Isso pode ser um problema momentâneo ou indicar o fim de dados esparsos.
                    if current_empty_retry < retries_on_empty:
                        current_empty_retry += 1
                        logger.warning(
                            f"Resultados vazios para {symbol} mas next_url ({next_page_url}) existe. Tentativa {current_empty_retry}/{retries_on_empty} de seguir next_url."
                        )
                        url = (
                            next_page_url + f"&apiKey={self._api_key}"
                            if self._api_key not in next_page_url
                            else next_page_url
                        )
                        params = (
                            {}
                        )  # next_url geralmente contém todos os params necessários
                        attempt = (
                            0  # Resetar tentativas de erro 429/500 para a nova URL
                        )
                        logger.debug(
                            f"Esperando {self._request_delay_seconds}s antes da próxima requisição de página para {symbol}..."
                        )
                        # Usar o novo delay configurável
                        time.sleep(self._request_delay_seconds)
                        continue
                    else:
                        logger.warning(
                            f"Resultados vazios para {symbol} após {retries_on_empty} tentativas com next_url. Considerando fim dos dados para esta sub-janela."
                        )
                        break  # Sai do while loop

                # Sem resultados e sem next_url (ou após retries_on_empty)
                if not results:
                    logger.info(
                        f"Nenhum resultado encontrado para {symbol} na URL: {url}. Verifique o período e o símbolo."
                    )
                    break  # Sai do while loop

                for bar in results:
                    all_data.append(
                        {
                            # Usar fromtimestamp para datetime local, se for UTC use utcfromtimestamp
                            "date": datetime.fromtimestamp(bar["t"] / 1000.0),
                            "open": bar["o"],
                            "high": bar["h"],
                            "low": bar["l"],
                            "close": bar["c"],
                            "volume": bar["v"],
                        }
                    )

                url = (
                    next_page_url + f"&apiKey={self._api_key}"
                    if next_page_url and self._api_key not in next_page_url
                    else next_page_url
                )
                params = {}  # next_url geralmente contém todos os params necessários
                logger.debug(
                    f"Esperando {self._request_delay_seconds}s antes da próxima requisição de página para {symbol}..."
                )
                # Usar o novo delay configurável
                time.sleep(self._request_delay_seconds)
                attempt = 0  # Resetar tentativas de erro 429/500 para a nova URL
                current_empty_retry = 0  # Resetar retries de 'results' vazio

            # Captura erros de conexão, timeout, etc.
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Erro de requisição ao baixar dados do Polygon para {symbol}: {e} na URL: {url if 'url' in locals() else 'URL inicial'}"
                )
                if attempt < MAX_ATTEMPTS:
                    wait = (2**attempt) + (random.uniform(0, 1))
                    logger.warning(
                        f"Tentativa {attempt + 1}/{MAX_ATTEMPTS}. Esperando {wait:.2f}s antes de tentar novamente..."
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue
                else:
                    logger.error(
                        f"Falha definitiva de requisição após {MAX_ATTEMPTS} tentativas para {symbol}."
                    )
                    return pl.DataFrame([])  # Retorna DF vazio
            # Captura outros erros (ex: JSONDecodeError)
            except Exception as e:
                logger.error(
                    f"Erro inesperado ({type(e)}) ao processar dados do Polygon para {symbol}: {e} na URL: {url if 'url' in locals() else 'URL inicial'}"
                )
                # Para erros inesperados, pode ser melhor parar para este símbolo/janela.
                return pl.DataFrame([])

        if not all_data:
            logger.info(
                f"Nenhum dado coletado para {symbol} no período {start_date}-{end_date}"
            )
            return pl.DataFrame([])

        df = pl.DataFrame(all_data)
        # A conversão e ordenação são feitas em _download_data_in_range ou no final de _load_or_download_symbol
        # if "date" in df.columns and df.height > 0:
        #     df = df.sort("date")
        logger.info(
            f"Dados baixados em _download_symbol para {symbol} ({start_date}-{end_date}): {df.height} linhas."
        )
        return df

    def update_bars(self) -> Optional[MarketEvent]:
        if (
            self._global_time_index is None
            or self._global_time_index.is_empty()
            or (
                hasattr(self._global_time_index, "len")
                and self._current_time_idx_ptr >= self._global_time_index.len()
            )
            or (
                hasattr(self._global_time_index, "__len__")
                and self._current_time_idx_ptr >= len(self._global_time_index)
            )
        ):
            self.continue_backtest_flag = False
            return None
        current_global_dt = self._global_time_index[self._current_time_idx_ptr]
        self._current_datetime = current_global_dt
        for symbol in self._symbol_list:
            df = self._data[symbol]
            if hasattr(df, "height") and df.height == 0:
                self._latest_symbol_bars[symbol] = None
                continue
            bar_df = df.filter(pl.col("date") == current_global_dt)
            if hasattr(bar_df, "height") and bar_df.height > 0:
                self._latest_symbol_bars[symbol] = bar_df
            else:
                self._latest_symbol_bars[symbol] = None
        self._current_time_idx_ptr += 1
        self.continue_backtest_flag = self._current_time_idx_ptr < (
            self._global_time_index.len()
            if hasattr(self._global_time_index, "len")
            else len(self._global_time_index)
        )
        return MarketEvent(timestamp=current_global_dt)

    def get_latest_bars_df(self, symbol: str, N: int = 1) -> Optional[pl.DataFrame]:
        df = self._data[symbol]
        if self._current_datetime is None or (hasattr(df, "height") and df.height == 0):
            return None
        mask = df["date"] <= self._current_datetime
        filtered = df.filter(mask)
        if hasattr(filtered, "height") and filtered.height == 0:
            return None
        return filtered.tail(N)

    def get_latest_bars(self, symbol: str, N: int = 1) -> Optional[pl.DataFrame]:
        return self.get_latest_bars_df(symbol, N)

    def get_latest_bar_value(self, symbol: str, val_type: str) -> Optional[float]:
        bar_df = self._latest_symbol_bars.get(symbol)
        if bar_df is None or (hasattr(bar_df, "height") and bar_df.height == 0):
            return None
        val_type = val_type.lower()
        if val_type not in bar_df.columns:
            return None
        return bar_df[val_type][0]

    def get_latest_bar_datetime(self, symbol: str) -> Optional[datetime]:
        bar_df = self._latest_symbol_bars.get(symbol)
        if bar_df is None or (hasattr(bar_df, "height") and bar_df.height == 0):
            return None
        val = bar_df["date"][0]
        if hasattr(val, "to_pydatetime"):
            return val.to_pydatetime()
        return val

    def get_all_bars_between(
        self, start_date: datetime, end_date: datetime
    ) -> pl.DataFrame:
        dfs = []
        for symbol in self._symbol_list:
            df = self._data[symbol]
            if hasattr(df, "height") and df.height == 0:
                continue
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            filtered = df.filter(mask)
            if hasattr(filtered, "height") and filtered.height > 0:
                dfs.append(filtered)
        if dfs:
            return pl.concat(dfs)
        return pl.DataFrame([])

    def get_all_bars_in_period(self, symbol: str, period: str) -> pl.DataFrame:
        df = self._data[symbol]
        if hasattr(df, "height") and df.height == 0 or "date" not in df.columns:
            return pl.DataFrame([])
        mask = df["date"].dt.strftime("%Y-%m").str.contains(period)
        return df.filter(mask)
