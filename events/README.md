# Módulo Events

Este módulo define as diferentes classes de eventos usadas no sistema de backtesting para comunicação entre os componentes (DataHandler, Strategy, Portfolio, ExecutionHandler, RiskManager, etc).

## Estrutura do Módulo

- **`base.py`**
  - Define a classe base abstrata **`Event`**:
    - Atributos: `type` (EventType), `timestamp` (datetime)

- **`types.py`**
  - Define enums auxiliares:
    - **`EventType`**: MARKET, SIGNAL, ORDER, FILL, NEWS, REBALANCE, VOLATILITY
    - **`Direction`**: BUY, SELL, HOLD
    - **`OrderType`**: MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP

- **Eventos principais:**
  - **`market_event.py`**: `MarketEvent` (chegada de novos dados de mercado)
  - **`signal_event.py`**: `SignalEvent` (sinal de trading gerado pela estratégia)
  - **`order_event.py`**: `OrderEvent` (ordem de trading criada pelo portfólio)
  - **`fill_event.py`**: `FillEvent` (confirmação de execução de ordem)

- **Eventos especiais:**
  - **`news_event.py`**: `NewsEvent` (evento de notícia relevante)
  - **`rebalance_event.py`**: `RebalanceEvent` (evento de rebalanceamento de portfólio)
  - **`volatility_event.py`**: `VolatilityEvent` (evento de volatilidade extrema)

- **`__init__.py`**
  - Expõe todas as classes e enums principais do pacote para importação direta.

## Funcionalidades Desenvolvidas

- Sistema de eventos desacoplado para comunicação entre todos os componentes do backtesting.
- Suporte a eventos de mercado, sinais, ordens, preenchimentos, notícias, rebalanceamento e volatilidade extrema.
- Enumerações centralizadas para tipos de evento, direção e tipo de ordem.
- Todos os eventos possuem timestamp e podem ser facilmente estendidos.
- Métodos utilitários para representação textual e cálculo de custo de transação (em FillEvent).
- Compatível com filas de eventos (queues) para processamento assíncrono.

## Fluxo de Eventos Comum

1. **DataHandler** gera um `MarketEvent` quando novos dados estão disponíveis.
2. **Strategy** recebe o `MarketEvent` e, se as condições forem atendidas, gera um `SignalEvent`.
3. **Portfolio** recebe o `SignalEvent` e, com base nas regras, gera um `OrderEvent`.
4. **ExecutionHandler** recebe o `OrderEvent`, simula a execução e gera um `FillEvent`.
5. **Portfolio** recebe o `FillEvent` e atualiza as posições e o capital.

Eventos especiais como `NewsEvent`, `RebalanceEvent` e `VolatilityEvent` podem ser gerados por componentes específicos e processados conforme a lógica do sistema.

## Exemplos de Uso

### Criando um evento de mercado

```python
from events import MarketEvent
import datetime

market_event = MarketEvent(timestamp=datetime.datetime.now())
```

### Criando um sinal de compra

```python
from events import SignalEvent, Direction
import datetime

signal_event = SignalEvent(
    symbol="PETR4",
    direction=Direction.BUY,
    strength=0.8,
    timestamp=datetime.datetime.now()
)
```

### Criando uma ordem de mercado

```python
from events import OrderEvent, OrderType, Direction
import datetime

order_event = OrderEvent(
    symbol="PETR4",
    order_type=OrderType.MARKET,
    direction=Direction.BUY,
    quantity=100.0,
    timestamp=datetime.datetime.now()
)
```

### Criando um evento de preenchimento

```python
from events import FillEvent, Direction
import datetime

fill_event = FillEvent(
    symbol="PETR4",
    direction=Direction.BUY,
    quantity=100.0,
    fill_price=22.50,
    commission=1.25,
    exchange="B3",
    timestamp=datetime.datetime.now()
)
total_cost = fill_event.calculate_cost()
```

### Criando um evento de notícia

```python
from events import NewsEvent
import datetime

news_event = NewsEvent(
    timestamp=datetime.datetime.now(),
    source="Reuters",
    headline="Petrobras anuncia novo plano de investimentos",
    summary="A Petrobras divulgou hoje um novo plano estratégico para os próximos 5 anos..."
)
```

### Processando eventos em um loop de backtesting

```python
def backtest_loop(events_queue, data_handler, strategy, portfolio, execution_handler):
    while True:
        if data_handler.continue_backtest:
            data_handler.update_bars()
        else:
            break
        while not events_queue.empty():
            event = events_queue.get()
            if event.type == 'MARKET':
                strategy.calculate_signals(event)
                portfolio.update_timeindex(event)
            elif event.type == 'SIGNAL':
                portfolio.process_signal(event)
            elif event.type == 'ORDER':
                execution_handler.execute_order(event)
            elif event.type == 'FILL':
                portfolio.process_fill(event)
```

## API Pública

O módulo expõe as seguintes classes e enums principais para importação direta:
- `EventType`, `Direction`, `OrderType`
- `Event`, `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`, `NewsEvent`, `RebalanceEvent`, `VolatilityEvent`

---
