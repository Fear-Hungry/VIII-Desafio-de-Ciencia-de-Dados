import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

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
        """
        Cálculo vetorizado do RSI clássico:
          * delta = variação de preço
          * gain = delta positivo; loss = -delta quando delta < 0
          * avg_gain/loss = média móvel simples de gains e losses
          * rsi = 100 - 100/(1 + avg_gain/avg_loss)
        """
        delta     = close.diff()
        gain      = delta.clip(lower=0)
        loss      = -delta.clip(upper=0)
        avg_gain  = gain.rolling(period).mean()
        avg_loss  = loss.rolling(period).mean()
        rs        = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def next(self):
        # sinal de compra: RSI cruza para cima de rsi_buy
        if crossover(self.rsi, self.rsi_buy):
            self.buy()
        # sinal de venda: RSI cruza para baixo de rsi_sell
        elif crossover(self.rsi_sell, self.rsi):
            self.position.close()

# ========== Como rodar o backtest ==========

# 1. Carregue seus dados em um DataFrame 'df' com coluna 'Close' e índice datetime:
#    df = pd.read_parquet('seus_dados.parquet')  # ou pd.read_csv, etc.

# 2. Instancie e execute:
bt = Backtest(
    df,
    RsiHeuristic,
    cash=10_000,            # capital inicial
    commission=0.002,       # ex: 0.2% por operação
    trade_on_close=True,    # executa ordens sempre no fechamento da barra
)

stats = bt.run()
print(stats)   # exibe métricas de performance
bt.plot()      # gráfico interativo de equity e trades
