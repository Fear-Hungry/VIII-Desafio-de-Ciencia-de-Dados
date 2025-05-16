import pandas as pd
import numpy as np # Adicionado para np.nan
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Certifique-se de que seu arquivo CSV tenha colunas como:
# Date,Open,High,Low,Close,Volume
# A coluna 'Date' será o índice.
CSV_FILE_PATH = 'data/dados.csv'
PARQUET_FILE_PATH = 'data/dados.parquet'

# Carregar dados do CSV
try:
    df = pd.read_csv(CSV_FILE_PATH, parse_dates=['Date'], index_col='Date')

    # Garantir que os dados estão ordenados pelo índice (data)
    df.sort_index(inplace=True)

    required_columns = {'Open', 'High', 'Low', 'Close'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias faltando no CSV: {missing_columns}. Por favor, ajuste o arquivo ou os nomes das colunas.")

    if 'Volume' not in df.columns:
        print("Aviso: A coluna 'Volume' não foi encontrada. Algumas estratégias podem precisar dela, mas RsiHeuristic continuará.")

    # Salvar como Parquet para uso futuro ou se preferir trabalhar com Parquet
    df.to_parquet(PARQUET_FILE_PATH)
    print(f"Dados carregados de {CSV_FILE_PATH}, ordenados e salvos em formato Parquet: {PARQUET_FILE_PATH}")

except FileNotFoundError:
    print(f"Erro: O arquivo {CSV_FILE_PATH} não foi encontrado. Certifique-se de que ele exista no diretório 'data/'.")
    print("Por favor, crie um arquivo 'data/dados.csv' com seus dados de entrada (colunas: Date,Open,High,Low,Close,Volume)." )
    exit()
except ValueError as ve:
    print(f"Erro de valor ao processar o CSV: {ve}")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar ou processar o arquivo CSV: {e}")
    exit()

class RsiHeuristic(Strategy):
    # Parâmetros da estratégia (você pode ajustar ou otimizar depois)
    n_rsi       = 14    # período do RSI
    rsi_buy     = 30    # nível para gerar sinal de compra
    rsi_sell    = 70    # nível para gerar sinal de venda

    def init(self):
        # calcula o RSI e armazena em self.rsi
        self.rsi = self.I(self.compute_rsi, self.data.Close, self.n_rsi)

    @staticmethod
    def compute_rsi(close: pd.Series, period: int) -> pd.Series:
        delta     = close.diff()
        gain      = delta.clip(lower=0)
        loss      = -delta.clip(upper=0)
        avg_gain  = gain.rolling(period).mean()
        avg_loss  = loss.rolling(period).mean()

        # Lógica robusta para RSI quando avg_loss é 0:
        # Se avg_loss é 0, significa que não houve perdas no período.
        # Se avg_gain também é 0, RSI é indefinido (ou 0 ou 50 por convenção, usamos 0 aqui).
        # Se avg_gain > 0 e avg_loss é 0, RSI deve ser 100.
        # A atribuição redundante de `rs` foi removida daqui.

        rsi_values = np.where(avg_loss == 0,
                              np.where(avg_gain == 0, 0, 100),  # Caso avg_loss == 0
                              100 - (100 / (1 + avg_gain / avg_loss))) # Caso avg_loss != 0

        return pd.Series(rsi_values, index=close.index)

    def next(self):
        # sinal de compra: RSI cruza para cima de rsi_buy
        if crossover(self.rsi, self.rsi_buy):
            self.buy()
        # sinal de venda: RSI cruza para baixo de rsi_sell
        elif crossover(self.rsi_sell, self.rsi):
            self.position.close()

# ========== Como rodar o backtest ==========

# 1. Certifique-se de que seus dados CSV (data/dados.csv) estão prontos.
#    O script carregará automaticamente, ordenará e converterá para Parquet (data/dados.parquet).

# 2. Instancie e execute:
if 'df' in locals() and not df.empty: # Verifica se o DataFrame foi carregado
    bt = Backtest(
        df,                     # DataFrame com os dados carregados
        RsiHeuristic,
        cash=10_000,            # capital inicial
        commission=0.002,       # ex: 0.2% por operação
        trade_on_close=True,    # executa ordens sempre no fechamento da barra
    )

    stats = bt.run()
    print("\n--- Estatísticas do Backtest ---")
    print(stats)
    print("\n--- Detalhes dos Trades ---")
    print(stats['_trades'])
    # Adicionando uma visualização mais clara das métricas principais
    print(f"\nRetorno Final: {stats['Return [%]']:.2f}%")
    print(f"Buy & Hold Return: {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Número de Trades: {stats['# Trades']}")
    print(f"Taxa de Acerto: {stats['Win Rate [%]']:.2f}%")

    print("\nGerando gráfico do backtest...")
    bt.plot()      # gráfico interativo de equity e trades
else:
    print("O DataFrame está vazio ou não foi carregado. O backtest não será executado.")
