# Módulo Data Loader

Este módulo é responsável por carregar, transformar e gerenciar os dados de mercado necessários para as estratégias de backtesting. Ele unifica as funcionalidades anteriormente separadas nos pacotes `data_handler` e `data_loader` e oferece uma interface robusta, extensível e eficiente para manipulação de dados históricos.

## Estrutura do Módulo

- **`base.py`**
  - Define as interfaces abstratas para todos os manipuladores de dados:
    - **`DataHandler`** (ABC): Interface principal para backtesting avançado, com métodos para acesso incremental a dados históricos, suporte a eventos e múltiplos símbolos.
    - **`SimpleDataHandler`** (ABC): Interface simplificada para compatibilidade com código legado.

- **`csv_handler.py`**
  - Implementa **`CSVDataHandler`** (herda de `DataHandler`):
    - Carrega dados de múltiplos arquivos CSV (um por símbolo), usando Polars para alta performance e baixo uso de memória.
    - Suporte a streaming de grandes volumes de dados, tratamento de dados ausentes (métodos: `ffill`, `zeros`, `drop`), validação de integridade OHLC, detecção e tratamento de outliers.
    - Alinhamento temporal entre múltiplos ativos, filtragem por datas, mapeamento flexível de colunas.

- **`sql_handler.py`**
  - Implementa **`SQLDataHandler`** (herda de `DataHandler`):
    - Carrega dados históricos diretamente de bancos SQL, com suporte a múltiplos símbolos, tratamento de dados ausentes, validação de integridade e detecção de outliers.
    - Permite customização dos nomes de colunas, tabelas e métodos de preenchimento.

- **`loader.py`**
  - Implementa a classe **`DataLoader`** (independente da engine de eventos):
    - Carregamento genérico de dados de CSV e Parquet.
    - Pré-processamento: renomeação de colunas, conversão de tipos, ordenação, tratamento de nulos.
    - Adição de indicadores técnicos (se o módulo `indicators` estiver disponível).
    - Divisão treino/teste, criação de features/targets, salvamento em Parquet.
    - Útil para preparação de dados antes do backtest ou para análises exploratórias.

- **`utils/`**
  - Funções utilitárias para manipulação e transformação de dados:
    - **`resampling.py`**: Reamostragem de dados OHLCV para diferentes timeframes, alinhamento de múltiplos ativos, criação de datasets compostos, cálculo de retornos, criação de lags, adição de features temporais, etc.
    - **`adjustments.py`**: Ajustes retroativos de preços para eventos corporativos (splits, dividendos), vetorizados e eficientes.
    - **`__init__.py`**: Exporta as funções principais do pacote utils para fácil acesso.

- **`__init__.py`**
  - Expõe as principais classes e funções do pacote para importação direta: `DataHandler`, `SimpleDataHandler`, `CSVDataHandler`.

## Funcionalidades Desenvolvidas

- Interface unificada para acesso incremental e eficiente a dados de mercado.
- Suporte a múltiplas fontes: arquivos CSV, bancos SQL, e possibilidade de extensão para APIs.
- Streaming e processamento eficiente de grandes volumes de dados históricos.
- Tratamento robusto de dados ausentes, outliers e validação de integridade OHLC.
- Pré-processamento flexível, adição de indicadores técnicos, criação de features e targets.
- Utilitários para reamostragem, alinhamento temporal, ajustes de preços, cálculo de retornos e mais.
- Estrutura extensível para adicionar novas fontes de dados e utilitários.
- Cobertura de testes para cenários com dados faltantes, NaNs, diferentes frequências e eventos corporativos.

## Relação entre DataHandler, DataLoader e Utils

- **DataHandler**: Interface principal para o backtesting baseado em eventos. Fornece acesso incremental aos dados, gerencia múltiplos símbolos e envia eventos para a engine de backtesting.
- **DataLoader**: Classe independente para carregamento, pré-processamento e transformação de dados. Útil para preparação dos dados antes do backtest ou para análises exploratórias. Não é integrada diretamente ao motor de eventos.
- **Utils**: Funções auxiliares para reamostragem, ajustes de preços, criação de features, etc. Podem ser usadas tanto no pré-processamento quanto em pipelines customizadas.

## Exemplos de Uso

### Usando CSVDataHandler para backtesting

```python
from data_loader import CSVDataHandler

# Inicializa o manipulador de dados para backtesting
data_handler = CSVDataHandler(
    csv_dir="./data/stocks/",
    symbol_list=["PETR4", "VALE3"],
    start_date="2020-01-01",
    end_date="2020-12-31",
    fill_missing_method="ffill"  # Exemplo: preencher dados faltantes com o último valor válido
)

while data_handler.continue_backtest:
    data_handler.update_bars()
    latest_bars_df = data_handler.get_latest_bars_df("PETR4", N=10)
    last_close = data_handler.get_latest_bar_value("PETR4", "close")
```

### Usando DataLoader para pré-processamento

```python
from data_loader.loader import DataLoader

loader = DataLoader()
df = loader.lead_from_csv("./data/stocks/PETR4.csv")
df = loader.preprocess(df)
df = loader.add_technical_indicators(df, indicators=["SMA_20", "RSI_14", "MACD"])
train_df, test_df = loader.split_train_test(df, train_ratio=0.8)
loader.save_to_parquet(df, "./data/processed/PETR4_processed.parquet")
```

### Utilizando utilitários para reamostragem e ajustes

```python
from data_loader.utils import resample_ohlc, apply_splits, apply_dividends

# Reamostrando para timeframe diário
df_diario = resample_ohlc(df, time_col="date", timeframe="1D")

# Ajustando preços para splits e dividendos
df_ajustado = apply_splits(df_diario, splits_df)
df_ajustado = apply_dividends(df_ajustado, dividends_df)
```

## MarketstackDataHandler

Permite baixar e consumir dados históricos intraday (1-minuto, 5-min, etc) diretamente da API da Marketstack, para qualquer ativo suportado (ações globais, ETFs, índices, etc), de forma incremental e plug-and-play no backtesting.

### Parâmetros principais:
- `api_key`: sua chave de API da Marketstack
- `symbol_list`: lista de ativos (ex: ["AAPL", "PETR4.SA"])
- `interval`: intervalo dos candles ("1min", "5min", "15min", etc)
- `start_date`, `end_date`: período desejado (YYYY-MM-DD ou YYYY-MM-DDTHH:MM:SS)
- `cache_dir`: diretório para salvar os dados em Parquet/CSV (opcional, recomendado)
- `save_csv`: se True, salva também em CSV além do Parquet (padrão: False)

### Funcionamento do cache automático:
- Na primeira execução, baixa e salva os dados em Parquet (e opcionalmente CSV).
- Nas próximas execuções, usa o cache local se ele cobre o período solicitado.
- Se o período requisitado for maior que o cache, baixa apenas o período faltante e atualiza o arquivo.
- Junta, remove duplicatas e ordena os dados automaticamente.
- Reduz custos de API, acelera o processo e evita problemas de limite de requisições.

### Exemplo de uso incremental (plug-and-play):
```python
from data_loader import MarketstackDataHandler

data_handler = MarketstackDataHandler(
    api_key="SUA_API_KEY",
    symbol_list=["AAPL", "PETR4.SA"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    interval="1min",
    cache_dir="data/marketstack/",
    save_csv=True  # Opcional
)

while data_handler.continue_backtest:
    event = data_handler.update_bars()
    # Use os métodos get_latest_bars_df, get_latest_bar_value, etc.
```

Veja a documentação da Marketstack para detalhes de parâmetros: https://marketstack.com/documentation

## API Pública

O módulo expõe as seguintes classes e funções principais para importação direta:

- `DataHandler`, `SimpleDataHandler`, `CSVDataHandler`
- Utilitários: `resample_ohlc`, `get_timeframe`, `align_multiple_timeframes`, `apply_splits`, `apply_dividends`
- `DataLoader` (para pré-processamento e preparação de dados)

---
