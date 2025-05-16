import pandas as pd
import numpy as np # Adicionado para np.nan
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# O script tentará encontrar um arquivo data/dados.csv
# e processar suas colunas de forma flexível.
CSV_FILE_PATH = 'data/dados.csv'
# O nome do arquivo Parquet agora refletirá o nome do CSV original para clareza
PARQUET_FILE_PATH = CSV_FILE_PATH.replace('.csv', '.parquet')

# Carregar dados do CSV com tratamento flexível de colunas
try:
    df = pd.read_csv(CSV_FILE_PATH)

    # Normaliza todos os nomes para lowercase e remove espaços extras
    df.columns = df.columns.str.strip().str.lower()

    col_mapping = {}
    # Colunas esperadas (chave: nome padronizado backtesting.py; valor: lista de sinônimos em lowercase)
    expected_cols_data = {
        'Open':     ['open', 'abertura', 'preco_abertura'],
        'High':     ['high', 'maximo', 'preco_maximo', 'alta'],
        'Low':      ['low', 'minimo', 'preco_minimo', 'baixa'],
        'Close':    ['close', 'fechamento', 'preco_fechamento', 'ultimo'],
        'Volume':   ['volume', 'vol', 'quantidade', 'quant', 'vendas', 'negocios'] # Volume é opcional
    }

    found_cols = {}
    for std_name, variants in expected_cols_data.items():
        for v_lower in variants:
            if v_lower in df.columns:
                # Mapeia o nome original encontrado (já em lowercase) para o nome padrão Capitalizado
                col_mapping[v_lower] = std_name
                found_cols[std_name] = v_lower # Guarda qual variante foi encontrada
                break
        else:
            if std_name != 'Volume': # Volume é opcional
                raise ValueError(f"Coluna obrigatória '{std_name}' (ou sinônimos: {variants}) não encontrada no CSV.")

    # Detecta coluna de data automática (date, datetime, timestamp…)
    date_variants_lower = ['date', 'datetime', 'timestamp', 'time', 'data', 'hora']
    date_col_found_lower = next((c for c in df.columns if c in date_variants_lower), None)

    if date_col_found_lower is None:
        raise ValueError(f"Nenhuma coluna de data/hora (ex: {date_variants_lower}) encontrada no CSV.")
    else:
        # Converte e seta como índice
        df[date_col_found_lower] = pd.to_datetime(df[date_col_found_lower])
        df.set_index(date_col_found_lower, inplace=True)
        if date_col_found_lower in col_mapping: # Se a coluna de data também era uma das colunas de dados (improvável mas possível)
            del col_mapping[date_col_found_lower]

    # Renomeia as colunas de dados para o padrão do backtesting.py (Open, High, Low, Close, Volume)
    df.rename(columns=col_mapping, inplace=True)

    # Seleciona apenas as colunas que foram mapeadas para o padrão + o índice
    # Isso remove colunas extras que não são de data nem OHLCV
    final_columns = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
    df = df[final_columns]

    # Garantir que os dados estão ordenados pelo índice (data)
    df.sort_index(inplace=True)

    # Verificar se as colunas obrigatórias agora existem com o nome padrão
    required_after_rename = {'Open', 'High', 'Low', 'Close'}
    missing_after_rename = required_after_rename - set(df.columns)
    if missing_after_rename:
         # Este erro não deveria ocorrer se a lógica anterior funcionou
        raise ValueError(f"Erro interno: Colunas obrigatórias {missing_after_rename} não encontradas após o mapeamento.")

    if 'Volume' not in df.columns and 'Volume' in found_cols:
         print("Aviso: A coluna 'Volume' foi encontrada mas não mapeada corretamente. Verifique a lógica de mapeamento.")
    elif 'Volume' not in df.columns:
        print("Aviso: Nenhuma coluna de 'Volume' (ou sinônimos) foi encontrada ou mapeada. A estratégia RsiHeuristic continuará.")

    # Salvar como Parquet para uso futuro ou se preferir trabalhar com Parquet
    df.to_parquet(PARQUET_FILE_PATH)
    print(f"\nDados carregados de {CSV_FILE_PATH}, processados, colunas renomeadas para o padrão e salvos em {PARQUET_FILE_PATH}")
    print(f"Colunas detectadas e mapeadas: {found_cols}")
    print(f"Coluna de data/hora usada como índice: {date_col_found_lower}")
    print(f"Colunas finais no DataFrame para backtesting: {list(df.columns)}")
    print(f"Índice do DataFrame: {df.index.name}, Tipo: {df.index.dtype}\n")

except FileNotFoundError:
    print(f"Erro: O arquivo {CSV_FILE_PATH} não foi encontrado. Certifique-se de que ele exista no diretório 'data/'.")
    exit()
except ValueError as ve:
    print(f"Erro de valor ao processar o CSV: {ve}")
    exit()
except Exception as e:
    print(f"Ocorreu um erro inesperado ao carregar ou processar o arquivo CSV: {e}")
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
