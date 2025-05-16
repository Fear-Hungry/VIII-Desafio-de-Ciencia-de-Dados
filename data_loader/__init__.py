from .api_handler import PolygonDataHandler
from .base import DataHandler
from .csv_handler import CSVDataHandler
from .sql_handler import SQLDataHandler

"""
**Pacote Data Loader (`data_loader`)**

Este pacote é responsável por abstrair e fornecer acesso aos dados históricos
do mercado para o sistema de backtesting.

Ele define interfaces comuns (`DataHandler`) que as implementações
de carregadores de dados devem seguir e fornece implementações concretas
para diferentes fontes de dados (atualmente, arquivos CSV e SQL).

O objetivo é desacoplar a engine de backtesting dos detalhes específicos
de como os dados são armazenados e acessados.

Este pacote unifica as funcionalidades anteriormente separadas em `data_handler` e
`data_loader`, fornecendo uma interface mais limpa e consistente.

**Uso:**

```python
from backtesting.data_loader import CSVDataHandler

data_dir = 'path/to/your/csv/data'
symbols = ['AAPL', 'GOOG']
data_handler = CSVDataHandler(csv_dir=data_dir, symbol_list=symbols)

# A engine usará o data_handler para obter barras de dados
latest_bars = data_handler.get_latest_bars('AAPL', N=10)
market_event = data_handler.update_bars()
```

**Arquivos Principais:**

- `base.py`: Define as classes abstratas `DataHandler`.
- `csv_handler.py`: Implementa `DataHandler` para carregar dados de arquivos CSV.
- `sql_handler.py`: Implementa `DataHandler` para carregar dados de bancos SQL.
- `api_handler.py`: Implementa `DataHandler` para carregar dados de APIs (ex: Polygon).

**Este arquivo (`__init__.py`) serve para:**

- Expor as classes públicas do pacote para importação direta.
- Definir o que é considerado parte da API pública do pacote através de `__all__`.
"""

# Importa a classe base

# Importa implementações concretas

# Adicione outras implementações aqui (ex: DatabaseDataHandler, APIDataHandler)
# from .db_handler import DatabaseDataHandler

# Opcional: Controlar o que é exposto com __all__
__all__ = [
    "DataHandler",
    "CSVDataHandler",
    "SQLDataHandler",
    "PolygonDataHandler",
    # Adicione outras classes exportadas aqui
]
