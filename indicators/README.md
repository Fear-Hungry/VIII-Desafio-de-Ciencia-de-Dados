# Módulo Indicators

Este módulo contém a implementação de diversos indicadores técnicos utilizados em análises quantitativas e estratégias de trading.

## Estrutura do Módulo

- **`base.py`**
  - Define a classe abstrata **`Indicator`**:
    - Método abstrato: `calculate(data)`
    - Métodos utilitários: `get_column_name()`, `__str__()`, `__repr__()`

- **`types.py`**
  - Define tipos e enums relacionados aos indicadores:
    - **`IndicatorType`**: Enum com todos os tipos suportados (SMA, EMA, MACD, RSI, BB, ADX, ROC, STOCH, etc.)
    - **`IndicatorConfig`**: Classe de configuração para instanciar indicadores, valida parâmetros e gera nomes de coluna únicos.

- **Subdiretórios por categoria de indicadores:**
  - **`medias_moveis/`**: Médias móveis e derivados
    - `sma.py`: Simple Moving Average (SMA)
    - `ema.py`: Exponential Moving Average (EMA)
    - `macd.py`: Moving Average Convergence Divergence (MACD)
    - `adx.py`: Average Directional Index (ADX)
  - **`momento/`**: Indicadores de momento
    - `rsi.py`: Relative Strength Index (RSI)
    - `roc.py`: Rate of Change (ROC)
    - `stochastic_oscillator.py`: Stochastic Oscillator (%K, %D)
    - `cci.py`: Commodity Channel Index (CCI)
  - **`volatilidade/`**: Indicadores de volatilidade
    - `bollinger_bands.py`: Bollinger Bands (BBands)
    - `atr.py`: Average True Range (ATR)
    - `donchian_channels.py`: Donchian Channels
  - **`volume/`**: Indicadores baseados em volume
    - `obv.py`: On-Balance Volume (OBV)
    - `mfi.py`: Money Flow Index (MFI)
    - `vwap.py`: Volume Weighted Average Price (VWAP)
  - **`niveis/`**: Níveis de suporte/resistência
    - `fibonacci_retracement.py`: Fibonacci Retracement
  - **`tendencia/`**: Indicadores de tendência
    - `ichimoku_cloud.py`: Ichimoku Cloud

- **`__init__.py`**
  - Expõe as principais classes, enums e subpacotes para importação direta.

## Funcionalidades Desenvolvidas

- Implementação de dezenas de indicadores técnicos clássicos, organizados por categoria.
- Interface abstrata única para todos os indicadores (`Indicator`).
- Configuração flexível e validação automática de parâmetros via `IndicatorConfig`.
- Enumeração centralizada dos tipos suportados (`IndicatorType`).
- Suporte a múltiplos parâmetros e geração automática de nomes de coluna únicos.
- Fácil extensão: basta herdar de `Indicator` e implementar o método `calculate`.
- Compatível com DataFrames do Polars para alta performance.

## Indicadores Disponíveis

### Médias Móveis
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)

### Momento
- RSI (Relative Strength Index)
- ROC (Rate of Change)
- Stochastic Oscillator (%K, %D)
- CCI (Commodity Channel Index)

### Volatilidade
- Bollinger Bands (BBands)
- ATR (Average True Range)
- Donchian Channels

### Volume
- OBV (On-Balance Volume)
- MFI (Money Flow Index)
- VWAP (Volume Weighted Average Price)

### Níveis
- Fibonacci Retracement

### Tendência
- Ichimoku Cloud

## Exemplos de Uso

### Calculando uma Média Móvel Simples (SMA)

```python
import polars as pl
from indicators.types import IndicatorType, IndicatorConfig
from indicators.medias_moveis.sma import SMAIndicator

# Dados de exemplo
data = pl.DataFrame({
    'date': pl.date_range(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 1, 11), eager=True),
    'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
})

config = IndicatorConfig(IndicatorType.SMA, [5])
sma_indicator = SMAIndicator(config)
sma_values_df = sma_indicator.calculate(data)
data_with_sma = data.join(sma_values_df, on='date', how="left")
print(data_with_sma)
```

### Calculando Bandas de Bollinger

```python
import polars as pl
import datetime
from indicators.types import IndicatorType, IndicatorConfig
from indicators.volatilidade.bollinger_bands import BollingerBandsIndicator

data = pl.DataFrame({
    'date': pl.date_range(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 1, 11), eager=True),
    'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
})

config = IndicatorConfig(IndicatorType.BB, [5, 2.0])
bb_indicator = BollingerBandsIndicator(config)
bb_values_df = bb_indicator.calculate(data)
data_with_bb = data.join(bb_values_df, on='date', how="left")
print(data_with_bb)
```

### Calculando Múltiplos Indicadores

```python
import polars as pl
import datetime
from indicators.types import IndicatorType, IndicatorConfig
from indicators.medias_moveis.sma import SMAIndicator
from indicators.momento.rsi import RSIIndicator

data = pl.DataFrame({
    'date': pl.date_range(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 1, 11), eager=True),
    'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'high': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    'low': [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5],
    'volume': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
})

# SMA
sma_config = IndicatorConfig(IndicatorType.SMA, [5])
sma_indicator = SMAIndicator(sma_config)
sma_df = sma_indicator.calculate(data)
data = data.join(sma_df, on="date", how="left")

# RSI
rsi_config = IndicatorConfig(IndicatorType.RSI, [7])
rsi_indicator = RSIIndicator(rsi_config)
rsi_df = rsi_indicator.calculate(data)
data = data.join(rsi_df, on="date", how="left")

print(data)
```

## Como Criar Seu Próprio Indicador

Para criar um novo indicador, basta herdar de `Indicator` e implementar o método `calculate`:

```python
from indicators.base import Indicator
from indicators.types import IndicatorConfig
import polars as pl

class MeuIndicador(Indicator):
    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        # Sua lógica de cálculo aqui
        # Retorne um DataFrame com a coluna do indicador
        pass
```

## API Pública

O módulo expõe as seguintes classes, enums e subpacotes para importação direta:
- `Indicator`, `IndicatorConfig`, `IndicatorType`
- Subpacotes: `medias_moveis`, `momento`, `volatilidade`, `volume`, `niveis`, `tendencia`

---
