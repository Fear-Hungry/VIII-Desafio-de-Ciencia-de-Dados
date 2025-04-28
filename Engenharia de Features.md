## Engenharia de dados

### Divisão de tarefas

- [] Ingestão dos dados
    - [x] Conectar à API de dados escolhida
    - [x] Log de meta-dados (timestamp da coleta (5 em 5 minutos), parâmetros da API) para rastreabilidade.
- [] Armazenamento
    - [] Armazenar em um arquivo CSV (ou Parquet, que é mais otimizado que CSV).
- [] Limpeza & validação
    - [] Remover dados duplicados
    - [] Remover dados inconsistentes
    - [] Remover dados incompletos
    - [] Remover outliers
    - [] Normalização dos dados
- [] Transformação & Engenharia de features
    - [] Criação de novas features
    - [] Extração de features
    - [] Seleção de features
- [] Materialização & logging
- [] Preparação final para modelagem
    - [] Exportar feature store em formato compatível com o ambiente de treinamento de RL

### Recomendações
- Utilizar o formato Parquet para armazenamento dos dados, pois é mais otimizado que CSV, e a quantidade de dados é grande (muitooo grande!!).
- Utilização da biblioteca `polars` para manipulação de dados, pois é mais rápida que o `pandas` e tem uma API semelhante.
- Versionamento no GitHub, para manter o histórico de alterações e facilitar a colaboração entre os membros da equipe.

- FAÇAM A DOCUMENTAÇÃO, PFV!!
