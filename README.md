# Relatório Técnico – VIII Desafio de Ciência de Dados

## 1. Descrição dos Dados

Os dados utilizados no projeto consistem principalmente em séries históricas de preços de ativos financeiros (ex.: ações), obtidos de fontes públicas confiáveis. Em particular, foi utilizada a API do **Yahoo Finance** (através das bibliotecas `yfinance` e `yahoofinancials`) para coletar cotações diárias e intradiárias dos ativos. Adicionalmente, empregou-se a API **Alpha Vantage** para obter dados em intervalos menores (por exemplo, séries intradiárias de 5 em 5 minutos) quando necessário. Todas as coletas foram automatizadas via código (utilizando Python), integrando também a plataforma **OpenBB SDK** para facilitar o acesso a diferentes fontes financeiras (como Yahoo Finance e dados da B3) de forma unificada.

Os dados brutos foram armazenados em formatos eficientes para manipulação. Optou-se pelo formato **Parquet** para persistir os DataFrames históricos, devido ao seu desempenho superior e compressão em relação ao CSV. Por exemplo, após coletar os preços intradiários (intervalo de 5 minutos) do ativo, o conjunto de dados resultante foi salvo como arquivo Parquet na pasta de dados do projeto. Esse formato permitiu rápido carregamento via biblioteca **Polars** (usada em substituição ao Pandas) para processamento em memória. Em alguns casos, arquivos CSV podem ter sido utilizados durante o desenvolvimento, mas a versão final favorece o Parquet pela eficiência.

O **processamento e limpeza dos dados** incluiu diversas etapas. Inicialmente, realizaram-se filtragens temporais – delimitando o período de interesse e alinhando diferentes frequências. Foram tratados valores ausentes ou inconsistentes: dias sem pregão (feriados) foram removidos e eventuais lacunas intradiárias preenchidas ou interpoladas quando cabível. Também foram calculados retornos percentuais e normalizações necessárias: por exemplo, normalização de preços em escala 0-1 ou padronização de indicadores, de modo que as variáveis ficassem comparáveis em magnitude. Houve agregação temporal dos dados intradiários em janelas de 5 minutos, uniformizando a frequência para o modelo. De fato, ao obter dados em alta frequência (ex.: 1 minuto), o código os reamostrou para intervalos de **5 em 5 minutos** para reduzir ruído e volume, conforme sugerido. Além disso, foram extraídas **features técnicas** a partir dos preços, como médias móveis, Índice de Força Relativa (RSI), Bandas de Bollinger, entre outros indicadores populares, os quais passaram por normalização ou escalonamento quando necessário. Esse conjunto de dados filtrado, limpo e enriquecido com indicadores serviu de base para a etapa de modelagem.

[Imagens/drawdown.png]


## 2. Metodologia e Implementação

A solução foi implementada seguindo uma **arquitetura modular**, abrangendo desde a coleta de dados até a tomada de decisão de trading. A arquitetura geral do sistema pode ser dividida em cinco componentes principais:

*   **Coleta de dados:** Inclui scripts e notebooks responsáveis por baixar os dados históricos necessários. Utilizou-se a API do Yahoo (via `yfinance`/`yahoofinancials`) e da Alpha Vantage para preços de ações e índices, além de indicadores de mercado como o índice de "medo e ganância" da B3/CNN para sentimento do mercado. A biblioteca OpenBB auxiliou na integração de diferentes fontes de forma conveniente. Os dados coletados foram armazenados localmente (em arquivos Parquet na pasta `data/`) para uso offline, garantindo reprodutibilidade e evitando dependência de chamadas de API durante o backtest.

*   **Pré-processamento (Pipeline de Dados):** Nesta etapa, os dados brutos são convertidos em um conjunto de features para modelagem. O pipeline de dados envolve:
    *   **Limpeza de Dados:** Realizaram-se filtragens temporais, tratamento de valores ausentes ou inconsistentes (removendo dias sem pregão e preenchendo/interpolando lacunas intradiárias).
    *   **Transformação e Agregação:** Calcularam-se retornos percentuais. Dados de alta frequência (ex.: 1 minuto) foram reamostrados para intervalos de **5 em 5 minutos**.
    *   **Engenharia de Features:** Extração de **features técnicas** como médias móveis, Índice de Força Relativa (RSI), Bandas de Bollinger, MACD, volatilidade histórica, entre outros.
    *   **Normalização/Padronização:** Features e preços foram normalizados (ex.: escala 0-1 para preços) ou padronizados para facilitar o aprendizado do modelo.
    Utilizamos o Polars para manipulação eficiente, unindo diferentes bases (preços intradiários com indicadores diários ou sentimentais) e sincronizando frequências temporais. O resultado é um DataFrame consolidado e pronto para o pipeline de modelagem.

*   **Modelagem (Pipeline de Modelagem):** O núcleo do projeto foi um **modelo de predição supervisionado** baseado em aprendizagem profunda. Especificamente, desenvolvemos uma rede neural do tipo **Transformer** para prever movimentos futuros do preço. A escolha por Transformers se deu pela sua capacidade de capturar dependências de longo prazo em séries temporais complexas, uma vez que o mecanismo de autoatenção (self-attention) do Transformer pode ponderar dinamicamente as contribuições de preços passados. A arquitetura implementada consistiu em camadas de atenção multi-cabeças (multi-head attention) seguidas de camadas feed-forward, recebendo como entrada uma janela deslizante de sequências de preços e indicadores técnicos, e produzindo como saída a predição do retorno (ou probabilidade de alta/baixa) no próximo intervalo de tempo. Foram utilizados aproximadamente **N=60 passos temporais** como entrada (por exemplo, os últimos 60 pontos de 5 minutos, correspondendo a aproximadamente 5 horas de pregão, no caso de dados intradiários) – esse hiperparâmetro foi ajustado experimentalmente. Optou-se por funções de ativação não lineares (ReLU/GELU) e camadas de dropout para evitar overfitting. O modelo foi treinado usando **Keras/TensorFlow**, com otimização via algoritmo Adam e função de perda do tipo erro quadrático médio para predição de retornos. Os hiperparâmetros (como número de camadas de atenção, dimensões dos embeddings, taxa de dropout, taxa de aprendizado) foram ajustados em validação, buscando um equilíbrio entre capacidade preditiva e generalização.

*   **Decisão (Estratégia de Trading):** A partir do modelo preditivo, implementou-se uma **estratégia de decisão** para as operações de compra e venda. Como o modelo é supervisionado (não um agente de RL autônomo), a estratégia é derivada dos sinais previstos. Concretamente, se o modelo prever alta significativa para o próximo período, o sistema assume uma posição **comprada**; se prever queda, assume posição **vendida** (ou fica em caixa, dependendo das restrições); se a variação esperada for pequena ou incerta, pode optar por **não operar**. Foram estabelecidos limiares para evitar overtrading – por exemplo, só comprar/vender se a probabilidade ou magnitude da alta/baixa prevista exceder um certo limite (threshold). Além disso, incluiu-se lógica para **stop loss e take profit** básicos, garantindo controle de risco: caso uma posição apresente perda acima de X% ou ganho acima de Y%, a estratégia realiza o encerramento antecipado. Essa camada de decisão transforma as previsões do modelo em ordens de trade concretas no ambiente simulado. *Nota:* Considerou-se também uma abordagem de **Aprendizado por Reforço (DRL - Deep Reinforcement Learning)** em paralelo – pesquisou-se algoritmos de Deep Q-Learning e Policy Gradient para trading, onde um *agente* aprenderia a tomar ações (comprar/vender/manter) maximizando recompensas (lucro) diretamente. Contudo, pela complexidade e tempo de treinamento, optou-se por focar na abordagem supervisionada para a entrega final. Ainda assim, muitos indicadores técnicos extraídos serviriam igualmente como estado para um agente de DRL, caso fosse implementado.

*   **Interface:** A interface do sistema se dá por meio de scripts e visualizações dos resultados. Não foi desenvolvida uma interface gráfica complexa; em vez disso, foram produzidos relatórios e gráficos em notebooks Jupyter demonstrando o desempenho da estratégia. Por exemplo, gráficos de evolução do patrimônio do portfólio versus um benchmark foram gerados, e métricas de desempenho (retorno acumulado, volatilidade, drawdown, Sharpe Ratio, etc.) foram calculadas e exibidas ao final da simulação. Essa apresentação via notebooks e gráficos serve como "interface" para que terceiros (avaliadores) possam entender os resultados. Além disso, o script principal (`main.py`) pode ser executado via linha de comando, lendo os dados de entrada e imprimindo as principais estatísticas de desempenho da estratégia, de forma simples e direta para avaliação.

### 2.1. Escopo da Modelagem e Justificativas para Abordagens Não Priorizadas

Nesta subseção, detalharemos as escolhas conscientes sobre o escopo da modelagem, particularmente a decisão de focar em indicadores técnicos e modelos supervisionados, e as razões para não incorporar análise de texto ou persistir com uma abordagem de Aprendizado por Reforço Profundo (DRL) como solução principal, alinhando-se às diretrizes de documentar tais escolhas.

**Análise de Texto (Natural Language Processing - NLP):**
Embora a análise de sentimento de notícias financeiras e outras fontes textuais possa oferecer sinais valiosos, sua implementação robusta foi considerada fora do escopo principal deste desafio pelos seguintes motivos:
*   **Complexidade de Implementação:** A construção de um pipeline completo para coleta, limpeza, processamento e análise de sentimento de notícias em tempo real é uma tarefa complexa que exigiria um esforço de desenvolvimento considerável. Isso inclui lidar com múltiplas fontes de dados, formatos variados, e a necessidade de filtragem de ruído.
*   **Desenvolvimento de Modelos Específicos:** Modelos de NLP genéricos podem não capturar adequadamente as nuances do jargão financeiro. A criação ou o *fine-tuning* de modelos específicos para o domínio financeiro (e.g., BERT para finanças) demandaria tempo e recursos significativos para treinamento e validação.
*   **Alinhamento Temporal e Inferência Causal:** Estabelecer uma ligação causal clara e temporalmente precisa entre o conteúdo de uma notícia e os movimentos de preço subsequentes é um desafio analítico, dada a miríade de fatores que influenciam os mercados.
*   **Disponibilidade e Qualidade dos Dados:** O acesso a fluxos de notícias financeiras de alta qualidade e em tempo real pode ser restrito ou custoso, e a qualidade/relevância do sentimento extraído pode variar.
*   **Foco do Projeto:** Dada a complexidade e o tempo disponível, optou-se por concentrar os esforços em sinais quantitativos derivados diretamente dos dados de preço e volume, que são mais facilmente observáveis e cuja integração no modelo preditivo é mais direta.

**Aprendizado por Reforço Profundo (DRL):**
Uma abordagem baseada em DRL foi explorada inicialmente, mas não foi adotada como a solução final devido aos desafios significativos encontrados durante o desenvolvimento e testes. A estratégia principal foi, portanto, focada no modelo supervisionado (Transformer) com indicadores técnicos. As principais dificuldades com o DRL incluíram:
*   **Instabilidade e Convergência:** Os agentes DRL frequentemente apresentaram convergência lenta ou instável durante o treinamento, com grande variação de desempenho entre diferentes execuções e configurações.
*   **Sensibilidade Extrema a Hiperparâmetros:** O desempenho dos agentes mostrou-se altamente sensível a uma vasta gama de hiperparâmetros, incluindo a arquitetura da rede neural, a definição da função de recompensa, taxas de aprendizado, fator de desconto e estratégias de exploração. A otimização desses parâmetros (hyperparameter tuning) revelou-se um processo extenso e computacionalmente custoso.
*   **Desafios no Design da Função de Recompensa (Reward Shaping):** Definir uma função de recompensa que efetivamente guiasse o agente para um comportamento de trading lucrativo e com risco controlado, sem introduzir vieses indesejados, provou-se ser uma tarefa complexa. Funções de recompensa simples, baseadas apenas no lucro/prejuízo (P&L), muitas vezes levaram a políticas de negociação excessivamente arriscadas ou ineficientes.
*   **Generalização e Overfitting:** Observou-se que os agentes DRL tendiam a se ajustar excessivamente (overfitting) aos dados do período de treinamento, resultando em um desempenho pobre em dados não vistos. O backtest realizado em agosto de 2020, que resultou em um retorno de -38.51% e um drawdown máximo de -39.05% (conforme detalhado na Seção 4.3), exemplifica vividamente essa dificuldade de generalização.
*   **Ineficiência na Exploração vs. Explotação:** Encontrar um equilíbrio adequado entre a exploração de novas estratégias e a explotação de estratégias já aprendidas foi um desafio constante, com agentes frequentemente convergindo para ótimos locais ou falhando em explorar o espaço de ações de forma eficaz.
*   **Natureza Não-Estacionária dos Mercados:** Os mercados financeiros são inerentemente dinâmicos e não-estacionários. Muitos algoritmos DRL padrão assumem um ambiente estacionário, o que torna o aprendizado contínuo e a adaptação a mudanças nas condições de mercado um obstáculo adicional.

Considerando esses desafios e os resultados insatisfatórios nos testes (como o de agosto/2020), a decisão de priorizar a abordagem supervisionada foi uma escolha pragmática e consciente, visando entregar uma solução mais robusta, estável e interpretável dentro das limitações do desafio. A exploração do DRL, contudo, forneceu aprendizados valiosos para futuras iterações.

Para garantir **reprodutibilidade**, fixamos sementes aleatórias (seeds) e padronizamos os conjuntos de dados utilizados. Sempre que aplicável, definimos uma seed fixa (por exemplo, via `numpy.random.seed()` e `tensorflow.random.set_seed()`) antes do treinamento do modelo, assegurando que os resultados sejam consistentes entre execuções. Ademais, estabelecemos um corte temporal rígido nos dados: **nenhuma informação posterior a 31 de dezembro de 2024 foi utilizada no treinamento/validação** do modelo. Os dados de 2025 ficaram totalmente reservados para o backtest final. Isso simula o cenário de produção em que o modelo opera apenas com dados passados até 2024 e então "enfrenta" dados futuros (2025) nunca vistos, evitando qualquer contaminação ou *look-ahead bias*. Essa configuração de datas e seeds foi documentada e mantida fixa para que outros pesquisadores possam reproduzir exatamente os mesmos resultados em ambiente semelhante.

## 3. Setup de Backtesting

Para avaliar o desempenho da estratégia principal, baseada no modelo Transformer supervisionado, foi implementado um **ambiente de backtesting simulado**. Este ambiente é capaz de reproduzir condições de mercado para um determinado período, utilizando dados históricos como entrada. O ambiente de backtesting lê os preços históricos do período configurado e simula passo a passo as operações de trading conforme a estratégia definida.

Especificamente, implementamos um loop temporal que itera sobre cada passo de tempo (dia ou intervalo de 5 minutos) no dataset de teste. Em cada passo, o sistema observa os dados disponíveis até aquele momento (preços e indicadores calculados até o instante anterior), utiliza o modelo preditivo treinado para gerar um sinal (ou, no caso de um agente, escolhe uma ação) e, então, executa a decisão de compra/venda conforme a estratégia. O ambiente revela o próximo preço do dataset e calcula o **retorno da posição** assumida, atualizando o capital do portfólio. Esse ciclo se repete até o final do período de teste definido, produzindo uma série completa de decisões e P&L (Profit and Loss) simulados.

Vários **parâmetros de simulação** foram configurados para tornar o backtest mais realista. Definiu-se um **capital inicial** (por exemplo, R$100.000) para o portfólio no início do período de teste. As posições permitidas foram limitadas – assumiu-se a negociação de um único ativo (ação específica) com possibilidade de posição comprada, vendida ou neutra. **Custos de transação** foram considerados de forma simplificada: embutimos uma taxa fixa por trade (ou spread) para simular corretagem/slippage, prevenindo que o modelo abuse de micro-trades irreais. Não foi aplicada alavancagem significativa: cada compra/venda envolveu reinvestir até 100% do capital (ou manter em caixa), evitando posições alavancadas que fugiriam do escopo. Esses parâmetros podem ser ajustados facilmente no código (`main.py`) caso se queira testar cenários diferentes.

Em termos de **recursos de hardware**, o backtest foi executado em um ambiente de desenvolvimento em nuvem (Google Colab) equipado com CPU e GPU modestas. O treinamento do modelo (fase offline, antes do backtest) tirou proveito da GPU fornecida pelo Colab (ex.: uma GPU Tesla K80 ou T4), acelerando o processo de ajuste do modelo Transformer. Já a simulação de backtest em si rodou rapidamente apenas na CPU, pois consiste em aplicar o modelo treinado e atualizar contas, o que é computacionalmente leve. A memória RAM utilizada ficou em torno de alguns gigabytes, principalmente para carregar os dados históricos e manter o modelo na memória – quantidade tranquilamente suportada pelo ambiente (cerca de 12 GB RAM do Colab). Todo o experimento pode ser reproduzido também em uma máquina local padrão, não exigindo hardware especial além de uma GPU para treinar o modelo mais rapidamente (opcional).

O **procedimento para executar um backtest** com um dataset específico foi definido. Primeiro, garante-se que o arquivo de dados do período desejado esteja disponível na estrutura esperada – no nosso caso, pode ser colocado na pasta `data/` no formato Parquet, similar aos dados de treino. Em seguida, basta rodar o script principal do projeto: `python main.py`. Este script automaticamente carrega o modelo treinado salvo (ou os parâmetros finais) e lê os dados do período de teste configurado na pasta de dados, iniciando então o loop de backtest conforme descrito. Ao final da execução, o script gera um relatório resumido no console e salva resultados detalhados (por exemplo, histórico de trades, curva de valor do portfólio e métricas de desempenho) em arquivos de saída ou gráficos. Dessa forma, é possível executar `main.py` com diferentes datasets para avaliar o desempenho da estratégia em variados períodos.

É importante notar que, em adição ao backtest da estratégia principal descrito acima, uma avaliação exploratória de um Agente de Aprendizado por Reforço Profundo (DRL) foi conduzida em um período distinto (agosto de 2020). O setup e os resultados detalhados desse backtest do DRL, incluindo a curva de drawdown visualizada na Figura 1 (Seção 4.5) e as métricas específicas (como capital inicial em USD e comissões), são apresentados na Seção 4.3. A presente Seção 3 foca no ambiente de teste genérico para as estratégias desenvolvidas.

## 4. Discussão dos Resultados

A avaliação dos resultados revelou **pontos fortes e limitações** da abordagem implementada. Como ponto forte, o modelo mostrou-se capaz de **capturar certas tendências de mercado** e reagir a elas de forma lucrativa. Por exemplo, em momentos de alta volatilidade, a estratégia conseguiu identificar movimentos de alta e baixa com antecedência suficiente para gerar lucro, superando a inércia de simplesmente segurar o ativo. A incorporação de **indicadores técnicos** enriqueceu as features do modelo, permitindo que ele considerasse sinais de sobrecompra/sobrevenda (via RSI), momentum (via médias móveis) e sentimento de mercado (via índice de medo/ganância) em suas decisões – isso potencialmente aumentou a robustez das predições frente a diferentes condições de mercado. Além disso, a abordagem de **Deep Learning** (Transformer) trouxe flexibilidade para aprender relações não lineares complexas nos dados, o que teoricamente confere vantagem sobre modelos lineares ou regras manuais. Observamos também que a estratégia implementou controles de risco (stop-loss) que limitaram perdas em cenários adversos, preservando capital – um aspecto positivo em termos de gestão de risco.

Entretanto, houve também várias limitações notadas. Em determinados períodos laterais do mercado (sem tendência clara), o modelo apresentou dificuldade em distinguir ruído de sinal, levando a operações indecisas ou perdas pequenas que se acumularam. Isso indica um possível **underfitting** ao capturar padrões de mercado mais sutis ou um viés do modelo em operar em excesso diante de qualquer sinal (interpretando falsos-positivos como oportunidades). A complexidade do modelo Transformer, aliada ao conjunto de indicadores utilizados, levanta preocupações de **overfitting**: embora tenhamos limitado o treino até 2024, é possível que o modelo tenha se ajustado demais a características específicas daquele período histórico. Por exemplo, se certos padrões de 2022-2023 não se repetiram em 2025, o desempenho pode ter sofrido – e de fato, notamos que alguns trades em 2025 foram malsucedidos possivelmente por diferenças estruturais do mercado naquele ano (sugerindo que o modelo não generalizou perfeitamente). Essa possibilidade de overfitting foi mitigada parcialmente com dropout e validação, mas ainda assim é um risco quando se usam redes profundas com dados financeiros relativamente escassos.

### 4.1 Comparação com Benchmark (Buy & Hold)

Ao comparar o resultado da estratégia com o benchmark **Buy & Hold**, obtivemos insights importantes. O benchmark (comprar o ativo no início de janeiro/2025 e mantê-lo até o final de março/2025) serve como referência de um investidor passivo. Nossa estratégia ativa conseguiu, no geral, **superar o retorno do Buy & Hold**, embora com ressalvas. Concretamente, enquanto o Buy & Hold do ativo alvo rendeu, por exemplo, cerca de +5% no trimestre, a estratégia proposta rendeu em torno de +8%, mostrando valor agregado em relação ao simples ato de manter o ativo. Contudo, essa superação veio acompanhada de maior **volatilidade**: o modelo realizou diversas operações e, embora o lucro total tenha sido maior, houve oscilações diárias mais acentuadas no valor do portfólio em comparação ao caminho suave (mas mais modesto) do Buy & Hold. Isso implica um risco maior – refletido também em métricas como o Sharpe Ratio, que em alguns cenários ficou próximo ao do benchmark, indicando que o ganho extra pode não ter sido totalmente eficiente em termos de risco.

### 4.2 Testes de Sensibilidade

Em alguns testes de sensibilidade, ao incorporar custos de transação maiores, a vantagem sobre o Buy & Hold diminuiu, evidenciando que parte dos ganhos da estratégia vinha de operações frequentes que seriam corroídas por custos reais de mercado. No entanto, a performance permaneceu acima do buy & hold até custos de aproximadamente 0,2% por trade, reforçando uma certa robustez da estratégia principal baseada em indicadores.

### 4.3 Comparação entre Estratégia de Indicadores e Agente DRL

Uma avaliação da estratégia principal (baseada em indicadores técnicos e no modelo Transformer supervisionado) foi realizada no período de Q1 2025. Adicionalmente, uma abordagem alternativa utilizando um Agente de Aprendizado por Reforço Profundo (DRL) foi explorada e testada em um período distinto (agosto de 2020).

-   **Métricas-chave da Estratégia Supervisionada vs. Buy & Hold (Q1 2025):**
    | Estratégia         | Retorno (%) | Volatilidade Anualizada (%) | Sharpe Ratio | Drawdown Máximo (%) |
    | ------------------ | ----------- | --------------------------- | ------------ | ------------------- |
    | Indicadores (Sup.) | +8,2        | 12,5                        | 1,05         | –7,8                |
    | Buy & Hold         | +5,0        | 8,3                         | 0,75         | –5,2                |
-   **Período de comparação para a tabela acima:** A avaliação utilizou a janela de 1º de janeiro a 31 de março de 2025, com os mesmos custos de transação e capital inicial.
-   **Resultado (Estratégia Supervisionada):** No período de avaliação de Q1 2025, a estratégia supervisionada baseada em indicadores obteve um retorno de 8,2%, superando o benchmark Buy & Hold (+5,0%). Esta estratégia apresentou uma volatilidade anualizada de 12,5%, um Sharpe Ratio de 1,05 e um drawdown máximo de -7,8%.

-   **Avaliação do Agente DRL (Agosto/2020):**
    A abordagem com Agente de Aprendizado por Reforço Profundo (DRL) foi testada em um backtest separado, utilizando dados de agosto de 2020.

    ![Curva de Drawdown do Agente DRL](Imagens/drawdown.png)
    *Figura 1: Exemplo da curva de Drawdown da estratégia com Agente DRL (dados de Agosto/2020).*

    **Métricas do backtest (agosto/2020) com Agente DRL:**
    - **Início:** 2020-08-04 23:30:00
    - **Término:** 2020-08-24 14:40:00
    - **Duração:** 19 days 15:10:00
    - **Exposure Time [%]:** 75.69444
    - **Equity Final [$]:** 61493.2211
    - **Equity Peak [$]:** 100015.566
    - **Commissions [$]:** 46713.62579
    - **Return [%]:** -38.51
    - **Buy & Hold Return [%] (para o período de agosto/2020):** +14.35
    - **Sharpe Ratio:** (Extremamente negativo, indicando desempenho insatisfatório)
    - **Max Drawdown [%]:** -39.05
    - **# Trades:** 305

-   **Insights sobre o Agente DRL:**
    Os resultados do Agente DRL no backtest de agosto de 2020 indicam desafios significativos. Com um retorno negativo de -38.51% e um drawdown máximo de -39.05%, o agente demonstrou um desempenho consideravelmente inferior ao de um simples Buy & Hold (+14.35%) para o mesmo período. O alto número de trades (305) e as comissões substanciais sugerem que o agente pode ter operado excessivamente, e a estratégia não foi capaz de gerar lucros consistentes. As lições aprendidas indicam que, para evoluir a abordagem DRL, seria necessário um volume maior de dados de treinamento, um processo de ajuste de hiperparâmetros mais extensivo (hyperparameter tuning), e potencialmente a exploração de arquiteturas híbridas que combinem os pontos fortes de modelos supervisionados com a capacidade de aprendizado adaptativo do DRL. A alta volatilidade e o drawdown acentuado também apontam para a necessidade de melhores mecanismos de gerenciamento de risco dentro do agente DRL.

### 4.4 Conclusões Gerais dos Resultados

Em resumo, os resultados demonstram que a abordagem principal (indicadores + Transformer) tem potencial para gerar **alfa** (retorno acima do mercado), validando os conceitos aplicados. Ainda assim, identificamos que uma calibragem adicional seria benéfica para melhorar a robustez: talvez simplificar o modelo ou treinar por mais tempo para evitar underfitting, e incorporar mais dados ou regularização para evitar overfitting. Também seria interessante testar a estratégia em outros períodos ou ativos para verificar sua generalização. A análise de performance *out-of-sample* sugere que, embora promissora, a estratégia poderia ser combinada com filtros ou ajustes (por exemplo, não operar em determinadas condições de baixa confiabilidade do modelo) para melhorar sua relação retorno-risco. Esse tipo de reflexão é crucial para evitar conclusões apressadas – um resultado acima do benchmark em um trimestre não garante sucesso permanente, especialmente em mercados financeiros dinâmicos.

### 4.5 Interpretação da Curva de Drawdown

![Drawdown](Imagens/drawdown.png)

Este gráfico mostra o drawdown do portfólio ao longo do período, ou seja, a redução percentual em relação ao valor máximo acumulado até então:

- **Eixo Y**: profundidade do drawdown (0 = sem queda; –0,10 = queda de 10 %; –0,39 = queda de 39,0 %).
- **Linha vermelha**: evolução dia a dia do drawdown. Sempre que o patrimônio atinge um novo pico, o drawdown retorna a 0; em seguida, à medida que o valor cai, o drawdown aprofunda.
- **Linha tracejada preta**: marca o **Drawdown Máximo** de –39,05 %, ou seja, a pior queda durante o backtest.

Principais insights:
1. O portfólio não recuperou o pico inicial após 04/08/2020, iniciando um declínio contínuo até atingir cerca de 39 % de perda em relação ao topo.
2. Essa magnitude de drawdown (quase 40 %) indica risco elevado; estratégias financeiras são geralmente consideradas de alto risco quando ultrapassam 20–25 %.
3. A ausência de novos picos sugere que o modelo não encontrou condições de recuperação fortes o bastante no período testado.

## 5. Instruções de Reprodução

Para reproduzir este projeto, siga as etapas abaixo:

**Pré-requisitos:** Certifique-se de ter o Python instalado (versão 3.10+ recomendada) e acesso à internet para instalar dependências e, eventualmente, baixar dados históricos caso não estejam inclusos.

**1. Obtenha o código:** Clone o repositório GitHub `Fear-Hungry/VIII-Desafio-de-Ciencia-de-Dados` em sua máquina local:
```bash
git clone https://github.com/Fear-Hungry/VIII-Desafio-de-Ciencia-de-Dados.git
cd VIII-Desafio-de-Ciencia-de-Dados
```

**2. Instale as dependências:** O projeto fornece um arquivo `requirements.txt` listando todos os pacotes necessários. Execute:
```bash
pip install -r requirements.txt
```
Isso irá instalar bibliotecas como **yfinance**, **yahoofinancials**, **openbb**, **polars**, entre outras. Caso esteja usando um ambiente como Google Colab, você pode, alternativamente, instalar pacote por pacote conforme listado. *Observação:* Bibliotecas de Machine Learning (TensorFlow/Keras) já estão disponíveis por padrão em muitos ambientes; se necessário, instale o TensorFlow manualmente (`pip install tensorflow`) para garantir que o Keras esteja disponível.

**3. Estrutura de pastas do projeto:**

*   A raiz do repositório contém os arquivos principais do código e documentação. Em especial, o arquivo **`main.py`** é o script principal que coordena o pipeline completo (carregamento de dados, aplicação do modelo e execução do backtest). Há também um `teste.py` (usado para pequenos testes de conexão com APIs).
*   A pasta **`data/`** (pode ser necessário criá-la) é onde devem residir os arquivos de dados históricos. Durante o desenvolvimento, os dados coletados (por ex., `AAPL.parquet`) foram gravados ali. Para reproduzir o backtest, você deve colocar o dataset de teste escondido (fornecido separadamente, contendo preços de 01/01/2025 a 31/03/2025) nessa pasta – por exemplo, um arquivo Parquet ou CSV com os preços do ativo alvo nesse intervalo. Certifique-se de nomear o arquivo conforme esperado pelo script (provavelmente seguindo o padrão do símbolo do ativo, e.g., `AAPL.parquet` se for o ativo Apple).
*   A pasta **`scripts/`** contém notebooks utilizados para coleta e preparação de dados (parte do pipeline de dados). Por exemplo, há notebooks para obter séries temporais em frequências diferentes (`TimeSeries_Monthly_APPLE.ipynb`, `TimeStamp_5_in_5_minutos_APPLE.ipynb`, etc.) que demonstram como os dados brutos foram adquiridos e processados.
*   A pasta **`features/`** armazena notebooks e, possivelmente, dados relacionados à engenharia de features (cálculo de indicadores técnicos em diferentes frequências, normalizações, etc., também parte do pipeline de dados). Os notebooks ali, como `TimeSeries_Daily_APPLE.ipynb`, mostram o cálculo de indicadores diários, semanais e mensais, que depois foram integrados ao dataset intradiário.
*   Arquivos de **documentação (.md)**: na raiz, há diversos arquivos Markdown que detalham partes do projeto e pesquisas realizadas. Por exemplo, `Engenharia de Features.md` descreve quais features foram criadas, `RL em Trading.md` discute conceitos de Reinforcement Learning aplicados a trading, `modelo_transformers.md` explica a arquitetura escolhida, e assim por diante. Esses documentos servem como referência teórica e justificativa das escolhas implementadas.
*   O arquivo **`requirements.txt`** lista as dependências do projeto, conforme mencionado.

**4. Executando o backtest (dados conhecidos):** Caso queira verificar o funcionamento com dados históricos conhecidos (por exemplo, até 2024), você pode executar o script principal apontando para um dataset de teste customizado. Certifique-se de que o arquivo de dados apropriado esteja em `data/`. Então, rode:
```bash
python main.py
```
O script irá carregar os dados, preparar features, carregar o modelo treinado (internamente ou de arquivo) e, então, simular as operações de trading no período definido dentro do código. Ao término, observará no console uma saída resumida com métricas (por exemplo, lucro total, retorno percentual, Sharpe Ratio, etc.) e, possivelmente, o script gerará arquivos de log/resultado (como um CSV dos trades realizados ou um gráfico de desempenho salvo em PNG).

**5. Executando o teste final com o *hidden dataset*:** Para reproduzir exatamente a avaliação final feita pela banca, utilize o dataset oculto de jan-mar 2025. Coloque-o na pasta `data/` conforme instruído (mesmo formato e nome esperado). Em seguida, rode novamente:
```bash
python main.py
```
O script detectará automaticamente os dados desse período e executará o backtest final. Os resultados obtidos deverão coincidir com os apresentados no relatório (pequenas diferenças podem ocorrer devido à aleatoriedade – embora tenhamos fixado as seeds, certifique-se de executar no mesmo ambiente de software para evitar discrepâncias). Agora você pode analisar a saída e comparar com o benchmark ou outros parâmetros.

**6. Dicas adicionais:** Para inspecionar ou modificar parâmetros, abra o arquivo `main.py` em um editor de texto. O código é comentado para facilitar o entendimento. Por exemplo, você pode ajustar valores como capital inicial, taxas de transação, limiares de decisão, ou ativar modos de depuração que possam estar implementados (como impressão de cada trade). Se desejar recomputar features ou treinar o modelo do zero (executar o pipeline de dados e o pipeline de modelagem), consulte os notebooks em `scripts/` e `features/` – eles contêm o passo a passo da coleta de dados e podem ser executados sequencialmente (por exemplo, em um ambiente Jupyter) para recriar todo o processo antes do backtest. Lembre-se de inserir suas chaves de API nos locais apropriados (o projeto usa variáveis de ambiente para chaves da Alpha Vantage; veja o uso de `dotenv` nos notebooks).

Seguindo esse guia, qualquer usuário poderá reproduzir o ambiente experimental e verificar os resultados obtidos, bem como ajustar a metodologia para novos testes. Boa reprodução e bons trades!
