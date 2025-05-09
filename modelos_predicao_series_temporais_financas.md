# Modelos de predições estatísticos
## Oque é?
 Predição de séries temporais é o processo de estimar valores futuros de uma variável com base em seus valores passados ao longo do tempo. Infelizmente não existe não existe um único “melhor modelo” para prever, mas sim modelos mais adequados dependendo do tipo de dado, horizonte de previsão e objetivos.

 ## Abaixo listei um conjunto de modelos eficazes e populares:
1. ARIMA/SARIMA
Uso em finanças:
* Previsão de preços de ativos ou indicadores econômicos

Ex: previsão do preço de fechamento de uma ação

Limitações:
* Não lida bem com volatilidade variável
* Supõe erro com variância constante
------
2. GARCH e variantes
* Modelagem da volatilidade dos retornos
* Previsão de Value at Risk (VaR) e gestão de risco
 
Variantes úteis:
* EGARCH: captura assimetrias (ex: "efeito alavancagem")
* GJR-GARCH: permite choques negativos influenciarem mais a volatilidade que os positivos
------
3. VAR (Vector AutoRegression)
Modela inter-relação entre variáveis financeiras, como:
* Juros
* Inflação
* Câmbio
* Retornos de ações

Aplicações:
* Análise de política monetária
* Estudo de impacto de choques econômicos
------
4. Modelos com variáveis exógenas (ARIMAX, GARCHX)

* Incorpora variáveis externas (ex: notícias, taxas, índices)
-----
5. LSTM (Long Short-Term Memory) – Rede Neural Recorrente
Vantagens:

* Captura padrões complexos e não-lineares
* Ótima para séries temporais com dependência de longo prazo
* Funciona bem com dados multivariados (ex: preço, volume, indicadores)

Uso comum:
Previsão de retornos diários, preços futuros ou direção de tendência
-----
6. Transformer (Time Series Transformer, Informer, etc.)
Vantagens:

* Supera LSTM em longas sequências
* Processa dados em paralelo (mais rápido para grandes volumes)
* Modela atenção temporal — identifica quais momentos passados são mais importantes

Uso comum:
Previsão multivariada em janelas longas (ex: 1 semana de preços)
-----
7. Hybrid Models (LSTM + GARCH, ARIMA + Rede Neural, etc.)
Vantagens:

* Combina capacidade estatística com inteligência de padrões não-lineares

Ex: ARIMA modela a parte linear e LSTM a parte não-linear
-----
8. XGBoost / LightGBM (Boosted Trees)
Vantagens:
* Extremamente eficiente com engenharia de features (ex: médias móveis, RSI, MACD)
* Bom com dados tabulares e pode usar variáveis externas (notícias, sentimentos, etc.)
-----
## Qual eu recomendaria nós comerçarmos? 
Bem eu recomendaria: Arima, Sarima e Garch se tivermos mais tempo ou força computacional falaria para utilizarmos LSTM, XGBoost e Transformers.
