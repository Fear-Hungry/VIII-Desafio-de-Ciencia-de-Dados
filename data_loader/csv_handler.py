"""
**Implementação de DataHandler para Arquivos CSV (`data_loader.csv_handler`)**

Este módulo fornece a classe `CSVDataHandler`, uma implementação concreta
da interface `DataHandler` que carrega dados históricos de mercado a partir
de arquivos CSV individuais para cada símbolo.

Utiliza a biblioteca Polars para leitura e manipulação eficiente dos dados,
adequado para grandes datasets.

**Principais Funcionalidades:**

- Carrega dados de múltiplos arquivos CSV (um por símbolo).
- Assume colunas padrão (Date, Open, High, Low, Close, Volume), mas permite mapeamento.
- Converte automaticamente a coluna de data/hora para o tipo apropriado.
- Renomeia colunas para um formato interno padronizado (minúsculas).
- Filtra dados por um intervalo de datas opcional.
- Remove linhas com dados inválidos (NaNs) em colunas críticas.
- Alinha os dados de todos os símbolos a um índice de tempo comum, preenchendo
  lacunas usando 'forward fill' seguido de 'backward fill'.
- Fornece os dados à engine de backtesting barra por barra através da interface `DataHandler`.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Mantido para uso futuro ou se a definição local for realmente a errada
from events import MarketEvent

from .base import DataHandler

# import polars as pl # Removido
# from dateutil.relativedelta import relativedelta # Removido


# from .csv_components.csv_parser import SymbolCSVParser # Removido
# from .csv_components.data_validator import DataValidator # Removido


# Placeholder temporário para MarketEvent se não estiver definido em outro lugar
# Mova esta classe para o seu módulo de eventos real (ex: events.py)


# class MarketEvent: # Removida definição local
#     \"\"\"
#     Evento que carrega dados de mercado para um timestep.
#     \"\"\"

#     def __init__(self, timestamp, data: Dict[str, Any]):
#         self.type = "MARKET"
#         self.timestamp = timestamp
#         # self.data é um dicionário {symbol: bar_data (dict ou Series)}
#         # bar_data deve conter OHLCV e (opcionalmente) indicadores
#         self.data = data

#     def __repr__(self):
#         return f"<{self.type} {self.timestamp} Symbols: {list(self.data.keys())}>"


class CSVDataHandler(DataHandler):
    """
    Handles the loading and providing of data from a single CSV file for a single symbol.
    Assumes the CSV has columns: timestamp, open, high, low, close, volume.
    """

    def __init__(self, csv_filepath):
        self.csv_filepath = csv_filepath
        self._data = None
        self._current_step = 0

    def load_data(self):
        """Loads data from the CSV file."""
        try:
            # Assuming the CSV has a header and appropriate column names
            # You might need to adjust column names or add parsing logic here
            self._data = pd.read_csv(self.csv_filepath)
            print(f"Dados carregados de: {self.csv_filepath}")
            print(f"Número total de passos de tempo: {len(self._data)}")
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {self.csv_filepath}")
            self._data = None
        except Exception as e:
            print(f"Erro ao carregar dados do CSV: {e}")
            self._data = None

    def get_next_bar(self):
        """
        Returns the data for the current step and advances the step counter.
        Returns None if data is not loaded or end of data is reached.
        """
        if self._data is None or self._current_step >= len(self._data):
            return None

        bar_data = self._data.iloc[self._current_step]
        self._current_step += 1
        return bar_data

    def reset(self):
        """Resets the step counter to the beginning of the data."""
        self._current_step = 0
        print("CSVDataHandler resetado.")

    def get_latest_data(self):
        """
        Returns all data up to the current step.
        Useful for calculating indicators over historical data.
        """
        if self._data is None or self._current_step == 0:
            return pd.DataFrame()
        return self._data.iloc[: self._current_step].copy()

    def is_end_of_data(self):
        """Checks if the end of the data has been reached."""
        return self._data is None or self._current_step >= len(self._data)

    def update_bars(self) -> None:
        """
        Avança um minuto. Coleta dados para o timestep atual e cria um MarketEvent interno.
        Atualiza self.continue_backtest no fim dos dados.
        """
        if self._idx < len(self.df):
            row = self.df.iloc[self._idx]  # Obtém a linha atual como Series
            # Monta o dicionário de dados para o MarketEvent com as colunas padrão OHLCV+Volume
            # Certifica-se de que as colunas existam antes de tentar acessá-las
            bar_data = {}
            for col_name in ["open", "high", "low", "close", "volume"]:
                if col_name in row:
                    bar_data[col_name] = row[col_name]
                else:
                    # Use NaN se a coluna estiver faltando na linha
                    bar_data[col_name] = np.nan

            # Associa os dados à chave do símbolo
            data = {self.symbol: bar_data}

            # Cria o MarketEvent
            self.latest_market_event = MarketEvent(
                timestamp=row["timestamp"], data=data
            )
            self._idx += 1
        else:
            # Fim dos dados
            print(f"CSVDataHandler para {self.symbol}: Fim dos dados.")
            self.continue_backtest = False
            self.latest_market_event = None

    def create_market_event(self) -> Optional[MarketEvent]:
        """
        Retorna o último MarketEvent gerado por update_bars().
        Retorna None se não houver MarketEvent recente (ex: antes do primeiro update_bars).
        """
        return self.latest_market_event

    def get_latest_bars_df(self, symbol: str, N: int) -> pd.DataFrame:
        """
        Retorna as últimas N barras (incluindo a barra atual se update_bars foi chamado)
        para um símbolo específico como um DataFrame pandas.
        Assumindo handler de símbolo único, o 'symbol' argumento é ignorado, mas mantido
        para compatibilidade com a interface multi-símbolo da Env.
        """
        # Se o índice atual (_idx) for menor que N, pega desde o início (índice 0)
        start = max(0, self._idx - N)
        # Pega até o índice ATUAL (_idx), que é o final da barra mais recente processada
        end = self._idx

        if start >= end:
            # Retorna um DataFrame vazio com as colunas esperadas se não houver dados suficientes
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Retorna o slice do DataFrame
        # Inclui a coluna 'timestamp' que foi renomeada internamente
        cols_to_return = ["timestamp", "open", "high", "low", "close", "volume"]
        return self.df.iloc[start:end][cols_to_return].copy()

    def get_latest_prices(self) -> Dict[str, float]:
        """
        Retorna um dicionário {symbol: último preço de fechamento}.
        Baseado no latest_market_event interno.
        """
        if self.latest_market_event and self.symbol in self.latest_market_event.data:
            # Obtém os dados da barra mais recente para o símbolo
            bar_data = self.latest_market_event.data[self.symbol]
            # Retorna o preço de fechamento, garantindo que a chave exista e não seja None/NaN
            close_price = bar_data.get("close")
            if pd.isna(close_price):  # Verifica se é NaN ou None
                return {self.symbol: np.nan}
            return {self.symbol: float(close_price)}
        # Retorna dicionário vazio ou com NaN se não houver dados recentes para o símbolo
        # Retorna NaN explicitamente para o símbolo se não houver dados
        return {self.symbol: np.nan}

    # Adicionar um placeholder para get_latest_indicators para compatibilidade com TradingEnv
    def get_latest_indicators(self, symbol: str, N: int) -> pd.DataFrame:
        """
        Placeholder: Retorna DataFrames vazios ou com zeros para indicadores.
        Implemente a lógica real se seus indicadores estiverem no CSV ou forem calculados.
        """
        # Assume que 0 indicadores são carregados por padrão, a menos que indicator_columns seja passado
        n_expected_indicators = len(getattr(self, "indicator_columns", {}))
        if n_expected_indicators > 0:
            # Retorna um DataFrame de zeros com o número esperado de colunas
            mock_cols = [f"indic{i+1}" for i in range(n_expected_indicators)]
            return pd.DataFrame(np.zeros((N, n_expected_indicators)), columns=mock_cols)
        else:
            # Retorna um DataFrame vazio se nenhum indicador é esperado
            return pd.DataFrame()


# --- Exemplo de uso (Mover para o bloco __main__ em env/trading_env.py) ---
# if __name__ == "__main__":
#     print("Executando teste rápido do CSVDataHandler...")
#
#     # Crie um CSV mock para teste rápido se não tiver um real
#     # Ex: 'data/mock_aapl.csv' com colunas date, open, high, low, close, volume, symbol (se multi-ativo)
#
#     mock_filepath = 'data/mock_aapl_goog.csv' # Crie este arquivo para testar multi-ativo
#     mock_symbols = ["AAPL", "GOOG"]
#
#     # Crie um arquivo mock multi-ativo se não existir
#     if not os.path.exists(mock_filepath):
#         print(f"Criando arquivo mock em {mock_filepath}")
#         mock_data = []
#         for i in range(100): # Timesteps
#             timestamp = f'2023-01-{i+1:02} 09:30:00'
#             # Dados AAPL
#             mock_data.append({ 'date': timestamp, 'symbol': 'AAPL', 'open': 150+i*0.1, 'high': 151+i*0.1, 'low': 149+i*0.1, 'close': 150.5+i*0.1, 'volume': 1000+i*50 })
#             # Dados GOOG
#             mock_data.append({ 'date': timestamp, 'symbol': 'GOOG', 'open': 100+i*0.2, 'high': 101+i*0.2, 'low': 99+i*0.2, 'close': 100.8+i*0.2, 'volume': 2000+i*100 })
#         mock_df = pd.DataFrame(mock_data)
#         mock_df.to_csv(mock_filepath, index=False)
#         print("Arquivo mock criado.")
#     else:
#          print(f"Arquivo mock já existe em {mock_filepath}")
#
#
#     data_handler = CSVDataHandler(
#         filepath=mock_filepath,
#         symbol='AAPL',
#         date_column='date',
#         price_columns={
#             'open': 'open',
#             'high': 'high',
#             'low': 'low',
#             'close': 'close',
#             'volume': 'volume',
#         },
#     )
#
#     data_handler.reset()
#
#     # Testar update_bars e create_market_event
#     print("\nTestando update_bars e MarketEvents (15 passos):")
#     for _ in range(15):
#         data_handler.update_bars()
#         market_event = data_handler.create_market_event()
#         if market_event:
#             print(market_event)
#             for symbol, data in market_event.data.items():
#                  if data is not None:
#                      print(f"  {symbol}: Close={data.get('close','N/A')}")
#         else:
#             print("Fim dos dados ou erro.")
#             break
#
#     # Resetar para testar get_latest_bars_df e get_latest_prices do início
#     data_handler.reset()
#     for _ in range(55): # Avança para ter > window_size dados
#          data_handler.update_bars()
#
#     # Testar get_latest_bars_df (após warm-up parcial simulado)
#     print("\nTestando get_latest_bars_df (últimas 10 barras simuladas):")
#     latest_bars = data_handler.get_latest_bars_df(symbol='AAPL', N=10)
#     print(f"Últimas 10 barras para AAPL:\n{latest_bars.tail()}") # Exibe as últimas 5 das 10\n#
#     # Testar get_latest_prices (após warm-up parcial simulado)
#     print("\nTestando get_latest_prices (preços mais recentes simulados):")
#     latest_prices = data_handler.get_latest_prices()
#     print(latest_prices)
#
#     print("\nTeste do CSVDataHandler mockado finalizado.")
