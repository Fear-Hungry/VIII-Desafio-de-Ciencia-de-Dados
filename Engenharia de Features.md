## Engenharia de dados

### Divisão de tarefas

- [] Ingestão dos dados
    - [] Conectar à API de dados escolhida
    - [] Log de meta-dados (timestamp da coleta (5 em 5 minutos), parâmetros da API) para rastreabilidade.
- [] Armazenamento
    - [] Armazenar em um arquivo CSV (ou Parquet, que é mais otimizado que CSV).
- [] Limpeza e validação
    - [] Remover dados duplicados
    - [] Remover dados inconsistentes
    - [] Remover dados incompletos
    - [] Remover outliers
    - [] Normalização dos dados
- [] Particionamento temporal
    - [] Dividir os dados em janelas de tempo (ex: 5 minutos, 1 hora, 1 dia)
- [] Engenharia de features
    - [] Criação de novas features
    - [] Extração de features
    - [] Seleção de features
- [] Transformação de features
    - [] Normalização
    - [] Padronização
    - [] Escalonamento
- [] Seleção de features
- [] Materialização e logging
- [] Preparação final para modelagem
    - [] Exportar feature store em formato compatível com o ambiente de treinamento de RL

### Recomendações
- Utilizar o formato Parquet para armazenamento dos dados, pois é mais otimizado que CSV, e a quantidade de dados é grande (muitooo grande!!).
- Utilização da biblioteca `polars` para manipulação de dados, pois é mais rápida que o `pandas` e tem uma API semelhante.
- Versionamento no GitHub, para manter o histórico de alterações e facilitar a colaboração entre os membros da equipe.

- FAÇAM A DOCUMENTAÇÃO, PFV!!
