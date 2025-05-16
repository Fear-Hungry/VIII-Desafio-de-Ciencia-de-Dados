# | Dados & Dados

---

## **O que são indicadores de trading?**

**Indicadores técnicos** são ferramentas analíticas baseadas em cálculos matemáticos — que podem variar desde fórmulas complexas até análises gráficas simples, como o traçado de linhas de tendência. Esses indicadores são amplamente utilizados na **análise técnica** com o objetivo de identificar padrões e condições específicas no comportamento dos preços de um ativo financeiro.

Entre suas diversas funções, uma das mais importantes é a capacidade de detectar quando um ativo está sendo negociado em **condições de sobrecompra** ou **sobrevenda** . Essas situações ocorrem quando o preço se move excessivamente para cima ou para baixo em um curto período, sugerindo possíveis esgotamentos da tendência vigente e aumentando a probabilidade de correções ou reversões.

Os indicadores técnicos geralmente se baseiam em dados históricos do mercado, como os preços passados, o volume de negociação e, em alguns casos, o **interesse aberto** (open interest), especialmente em mercados futuros. A partir dessas informações, eles geram sinais que ajudam os traders a tomar decisões embasadas sobre entradas, saídas e gestão de risco.

O uso consistente desses indicadores permite ao trader desenvolver uma visão mais objetiva do mercado, auxiliando na identificação de oportunidades potenciais de negociação e contribuindo para estratégias mais sólidas e fundamentadas.

---

# **Lista dos melhores indicadores técnicos**

## 1. Média móvel (SMA)

A **média móvel** é um indicador estatístico amplamente utilizado na **análise técnica** para suavizar as flutuações de preços e auxiliar na identificação de tendências em ativos financeiros. O termo "móvel" deve-se ao fato de que esse indicador é constantemente atualizado, incorporando novos dados à medida que eles se tornam disponíveis e descartando os valores mais antigos.

Basicamente, a média móvel calcula a **média dos preços** — normalmente o preço de fechamento — de um determinado ativo financeiro ao longo de um período predeterminado, como 10, 20 ou 50 dias. Conforme o tempo avança, o cálculo é refeito, integrando as informações mais recentes e excluindo as mais antigas, o que confere ao indicador sua característica dinâmica e contínua atualização.

$$
\text{Média Móvel} = \frac{\text{Somatório dos valores de um período limitado}}{\text{Número de dados que possui nesse período}}
$$

---

## 2. Média móvel exponencial (EMA)

A **Média Móvel Exponencial** (**EMA** , do inglês *Exponential Moving Average* ) é um indicador técnico que atribui pesos maiores aos preços mais recentes, dando-lhe maior sensibilidade às variações mais atuais do mercado. Diferentemente da **Média Móvel Simples (SMA)** , que atribui igual importância a todos os preços dentro do período analisado, a EMA prioriza as cotações dos períodos mais próximos da data atual, o que resulta em uma reação mais ágil às mudanças de preço.

Essa característica torna a EMA particularmente valiosa para traders que buscam identificar movimentos de tendência de forma mais rápida e com maior grau de precisão. Ao responder com maior prontidão a novas informações do mercado, a EMA pode ajudar na detecção precoce de reversões ou continuidades de tendências, sendo amplamente utilizada em estratégias de negociação de curto prazo e em sistemas de análise algorítmica.

$$
\alpha = \frac{2}{n + 1}
$$

$$
EMA_t = (P_t \cdot \alpha) + (\text{EMA}_{t-1} \cdot (1 - \alpha))
$$

---

## 3. Divergência de Convergência da média móvel (MACD)

O MACD (Moving Average Convergence Divergence) é um indicador técnico amplamente utilizado na análise de mercados financeiros, especialmente para medir a relação entre duas Médias Móveis Exponenciais (EMA) — normalmente as de 12 e 26 períodos. Essa diferença entre as EMAs gera uma série de sinais que ajudam traders a avaliar o comportamento do preço de um ativo.

$$
MACD = EMA_{12} - EMA_{26}
$$

$$
\text{Signal Line} = EMA_{9}(MACD)
$$

$$
\text{MACD Histogram} = MACD - \text{Signal Line}
$$

---

## 4. Índice de força relativa (RSI)

O **RSI (Relative Strength Index)** é um oscilador de momentum amplamente utilizado na análise técnica para medir a velocidade e a magnitude das variações de preço de um ativo financeiro. Desenvolvido por J. Welles Wilder Jr., o indicador apresenta valores em uma escala que varia entre **0 e 100** , permitindo aos traders avaliar se um ativo está em condições extremas de negociação.

Quando o RSI ultrapassa o nível **70** , isso indica que o ativo pode estar em uma zona de **sobrecompra** , ou seja, foi comprado de forma excessiva em um curto período, o que pode sugerir exaustão da tendência de alta e uma possível correção ou reversão. Por outro lado, quando o RSI cai abaixo do nível **30** , o ativo é considerado **sobrevendido** , indicando que foi vendido intensamente, o que pode sinalizar uma perda de força dos vendedores e uma potencial recuperação de preço.

$$
RS (Relative Strength) = \frac{U (altos)}{D(baixos)}
$$

$$
RSI = 100 - \left( \frac{100}{1 + RS} \right)
$$

```python
def calcular_rsi(preco, periodo):
  #calculando preços de fechamento consecutivos
  delta = preco.diff()

  #separando os ganhos das perdas
  ganhos = delta.clip(lower=0)
  perdas = -delta.clip(upper=0)

  #calculando a media de ganhos e perdas
  media_ganhos = ganhos.ewm(span=periodo, adjust=False).mean()
  media_perdas = perdas.ewm(span=periodo, adjust=False).mean()

  #calculando o rs e o rsi
  rs = media_ganhos / media_perdas
  rsi = 100 - (100 / (1 + rs))
  return rsi
```

## 5. SAR parabólico (PSAR)

O PSAR (Parabolic Stop and Reverse) é um indicador técnico amplamente utilizado na análise de mercados financeiros para identificar potenciais reversões na tendência do preço. Ele é representado graficamente por pontos posicionados acima ou abaixo das velas no gráfico de preços. Quando esses pontos aparecem abaixo das velas, o indicador sinaliza uma tendência de alta, sugerindo que o ativo está em movimento ascendente; por outro lado, quando os pontos estão localizados acima das velas, isso indica uma tendência de baixa, refletindo um movimento descendente no preço do ativo.

Além de auxiliar na identificação de tendências, o PSAR também é muito empregado para estabelecer níveis dinâmicos de *stop-loss* , permitindo ao investidor proteger seus lucros ou limitar perdas conforme o preço se move a favor da operação. Sua característica "parabólica" faz com que os pontos do indicador acelerem à medida que a tendência se desenvolve, ajustando-se dinamicamente ao comportamento do mercado. No entanto, como qualquer ferramenta de análise técnica, o PSAR pode gerar sinais falsos em períodos de baixa volatilidade ou em mercados laterais, devendo ser utilizado em conjunto com outros indicadores ou métodos para maior eficácia na tomada de decisão.

$$
\text{Para tendência de alta:} \\PSAR(t) = PSAR(t-1) + A \cdot \left( MAX(t-1) - PSAR(t-1) \right) \\[1em]
\text{Para tendência de baixa:} \\PSAR(t) = PSAR(t-1) - A \cdot \left( PSAR(t-1) - MIN(t-1) \right) \\[2em]

$$

- **PSAR(t)**: Valor atual do Parabolic SAR
- **PSAR(t−1)**: Valor anterior do Parabolic SAR
- **MAX(t−1)**: Maior preço (máxima) do período anterior
- **MIN(t−1)**: Menor preço (mínima) do período anterior
- **A**: Fator de aceleração, normalmente começa em 0,02 e pode ir até 0,2

```python
import pandas as pd
import numpy as np

def calculate_psar(df, af=0.02, af_max=0.2):
    high = df['High'].values
    low = df['Low'].values
    length = len(df)

    if length < 2:
        raise ValueError("DataFrame deve ter pelo menos 2 linhas para calcular o PSAR")

    psar = np.zeros(length)
    bull = True
    ep = high[0]  # Definido diretamente
    af_val = af

    # Determina a direção inicial com base nos dois primeiros candles
    if high[1] > high[0]:
        bull = True
        psar[0] = low[0]
        ep = high[0]
    else:
        bull = False
        psar[0] = high[0]
        ep = low[0]

    for i in range(1, length):
        prev_psar = psar[i - 1]

        if bull:
            psar[i] = prev_psar + af_val * (ep - prev_psar)
            psar[i] = min(psar[i], low[i - 1], low[i])

            if low[i] < psar[i]:  # Reversão para baixa
                bull = False
                psar[i] = ep
                ep = low[i]
                af_val = af
        else:
            psar[i] = prev_psar + af_val * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1], high[i])

            if high[i] > psar[i]:  # Reversão para alta
                bull = True
                psar[i] = ep
                ep = high[i]
                af_val = af

        # Atualiza EP e AF se a tendência continuar
        if bull:
            if high[i] > ep:
                ep = high[i]
                af_val = min(af_val + af, af_max)
        else:
            if low[i] < ep:
                ep = low[i]
                af_val = min(af_val + af, af_max)

    df['PSAR'] = psar
    return df
```

## 6. Índice direcional médio (ADX)

O **Índice Direcional Médio (ADX)** é um indicador técnico amplamente utilizado para **medir a força de uma tendência**, independentemente de ela ser ascendente ou descendente. Diferentemente de outros indicadores que apontam direção, o ADX se concentra unicamente na **intensidade da tendência**, ajudando o trader a determinar se o movimento em curso possui embasamento suficiente para ser considerado válido .

O valor do ADX varia entre 0 e 100, sendo que **valores acima de 25** costumam indicar a presença de uma **tendência forte**, enquanto **valores abaixo de 20** sugerem um **mercado sem definição clara de direção**, possivelmente em congestão ou movimento lateral . Quanto maior o valor do ADX (especialmente acima de 25), mais forte é considerada a tendência em andamento — o que pode ser útil para confirmar a validade de sinais de entrada ou para evitar operações contrárias à tendência principal .

Além da linha principal do ADX, o indicador também inclui duas linhas complementares:

- **+DI (Positive Directional Indicator)** – que mede a força do movimento ascendente;
- **–DI (Negative Directional Indicator)** – que mede a força do movimento descendente.

A interação entre essas duas linhas permite identificar mudanças no **sentido da tendência**. Por exemplo, quando a linha **+DI cruza acima do –DI**, isso pode sinalizar o início de uma **tendência de alta**, enquanto o cruzamento inverso pode indicar uma possível **mudança para uma tendência de baixa**. Essas informações, combinadas com a leitura do ADX sobre a força da tendência, oferecem ao trader um quadro mais completo para a tomada de decisão .

Vale ressaltar que o ADX não deve ser usado isoladamente, pois, embora seja eficaz na medição da intensidade da tendência, ele não indica diretamente a direção do preço. Assim, seu uso combinado com outros indicadores ou ferramentas de análise pode aumentar sua eficiência e evitar sinais enganosos, especialmente em mercados de baixa volatilidade ou sem tendências definidas .

$$
+DM_t =\begin{cases}\text{Alta}_t - \text{Alta}_{t-1}, & \text{se } \text{Alta}_t - \text{Alta}_{t-1} > \text{Baixa}_{t-1} - \text{Baixa}_t \text{ e } > 0 \\0, & \text{caso contrário}\end{cases}
$$

$$
-DM_t =\begin{cases}\text{Baixa}_{t-1} - \text{Baixa}_t, & \text{se } \text{Baixa}_{t-1} - \text{Baixa}_t > \text{Alta}_t - \text{Alta}_{t-1} \text{ e } > 0 \\0, & \text{caso contrário}\end{cases}
$$

$$
TR_t = max(Alta_t - Baixa_t, |Alta_t - Fech_{t-1}|, |Baixa_t - Fech_{t-1}|)
$$

$$
+DI_t = 100 \cdot \frac{+DM_t}{TR_t}
$$

$$
-DI_t = 100 \cdot \frac{-DM_t}{TR_t}
$$

$$
DX_t = 100 \cdot \frac{ |+DI_t - (-DI_t)| }{ +DI_t + (-DI_t) }
$$

$$
ADX_t = \text{Média dos últimos } n \text{ valores de } DX_t
$$

**Legenda das Variáveis:**

- **Altaₜ**: Preço máximo do período atual
- **Baixaₜ**: Preço mínimo do período atual
- **Fechₜ₋₁**: Preço de fechamento do período anterior
- **+DMₜ** e **−DMₜ**: Movimentos direcionais positivos e negativos
- **TRₜ**: Faixa verdadeira (True Range)
- **+DIₜ**, **−DIₜ**: Indicadores direcionais positivos e negativos
- **DXₜ**: Índice Direcional (Directional Index)
- **ADXₜ**: Média suavizada de **DXₜ**, que indica a força da tendência

## 7. Oscilador estocástico

O **Oscilador Estocástico** é um indicador técnico de momentum amplamente utilizado na análise de mercados financeiros para avaliar o movimento do preço de um ativo em relação à sua faixa de preços histórica em um determinado período. Ele compara o **preço de fechamento mais recente** com a **mínima e a máxima registradas nesse intervalo**, permitindo identificar padrões que sugerem possíveis reversões no movimento do preço .

Este oscilador é representado graficamente por dois valores que variam entre **0 e 100**:

- A linha **%K**, que reflete o preço de fechamento atual em relação à faixa de preços dos últimos períodos analisados;
- E a linha **%D**, uma média móvel da linha %K, utilizada como linha de sinal para confirmar os movimentos detectados .

Um dos principais usos do Oscilador Estocástico é a identificação de **condições extremas de mercado**:

- Quando o valor do oscilador ultrapassa **80**, o ativo é considerado **sobrecomprado**, ou seja, pode estar em uma zona de excesso de compra e sujeito a uma correção ou reversão de curto prazo;
- Já quando o oscilador cai abaixo de **20**, o ativo é considerado **sobrevendido**, o que pode indicar uma possível perda de pressão vendedora e início de uma reversão positiva .

Além disso, o Oscilador Estocástico também ajuda a identificar **mudanças no momentum do preço**, antecipando potenciais inversões na tendência. Por exemplo, divergências entre o comportamento do preço e o do oscilador podem sinalizar fraqueza ou fortalecimento iminente no movimento atual . No entanto, assim como outros indicadores, ele deve ser utilizado em conjunto com outras ferramentas de análise técnica — como médias móveis, RSI ou análise gráfica — para evitar sinais falsos e aumentar a confiabilidade das operações .

$$
\%K_t = 100 \cdot \frac{P_t - \text{Low}_n}{\text{High}_n - \text{Low}_n}
$$

$$
\%D_t = \text{Média móvel simples de 3 períodos de } \%K_t
$$

**Legenda das Variáveis:**

- **%Kₜ**: Valor atual do Oscilador Estocástico
- **%Dₜ**: Média móvel de 3 períodos da linha %K (linha de sinal)
- **Pₜ**: Preço de fechamento do período atual
- **Lowₙ**: Mínimo dos últimos *n* períodos
- **Highₙ**: Máximo dos últimos *n* períodos
- **n**: Número de períodos utilizados na análise (normalmente 14)

```python
def oscilador_estocastico(df, n=14, smooth_k=3, smooth_d=3):
    # Cálculo do %K
    df['Low_min'] = df['Low'].rolling(window=n).min()
    df['High_max'] = df['High'].rolling(window=n).max()
    df['%K'] = ((df['Close'] - df['Low_min']) / (df['High_max'] - df['Low_min'])) * 100
    
    # Suavização do %K
    df['%K_smooth'] = df['%K'].rolling(window=smooth_k).mean()
    
    # Cálculo do %D
    df['%D'] = df['%K_smooth'].rolling(window=smooth_d).mean()
    
    # Remover colunas auxiliares
    df.drop(columns=['Low_min', 'High_max'], inplace=True)
    
    return df[['%K', '%D']]
```

## 8. Bandas de Bollinger

As **Bandas de Bollinger** são um indicador técnico amplamente utilizado na análise gráfica de mercados financeiros, composto por três linhas principais: uma **Média Móvel Simples (SMA)** posicionada no centro, e duas bandas externas — **superior e inferior** — que são calculadas com base no **desvio padrão** dos preços em torno dessa média central .

Por padrão, a banda central representa a **Média Móvel Simples de 20 períodos**, enquanto as bandas superior e inferior são obtidas somando e subtraindo, respectivamente, **duas vezes o desvio padrão** da Média Móvel central nesse mesmo intervalo . Essa estrutura permite que as bandas se **ajustem dinamicamente à volatilidade do mercado**: quando a volatilidade aumenta, as bandas se expandem; quando diminui, elas se contraem, criando uma espécie de "canal adaptável" ao redor do preço .

Esse comportamento adaptativo torna as Bandas de Bollinger uma ferramenta versátil para diferentes estratégias de negociação. Em ambientes de mercado lateral ou com baixa volatilidade, elas ajudam a identificar **níveis potenciais de suporte e resistência**, sendo úteis em estratégias de operação dentro do *range*. Já em momentos de forte movimentação, podem sinalizar **possíveis rompimentos** ou exaustão de tendências, especialmente quando o preço ultrapassa uma das bandas com volume significativo .

Além disso, o indicador também pode ser ajustado conforme a estratégia do trader, variando o número de períodos da média ou o múltiplo do desvio padrão, embora os parâmetros tradicionais sejam amplamente reconhecidos como eficazes na maioria dos cenários de mercado .

$$
\text{MM}_n = \text{Média Móvel Simples de } n \text{ períodos} \\[1em]
\text{Desvio}_n = \text{Desvio padrão dos últimos } n \text{ períodos} \\[1em]
\text{Banda Superior} = \text{MM}_n + k \cdot \text{Desvio}_n \\[1em]
\text{Banda Inferior} = \text{MM}_n - k \cdot \text{Desvio}_n
$$

- **MMₙ**: Média Móvel Simples de *n* períodos (geralmente 20)
- **Desvioₙ**: Desvio padrão dos preços nos últimos *n* períodos
- **k**: Fator de multiplicação do desvio padrão (normalmente 2)
- **Banda Superior**: Limite superior da banda
- **Banda Inferior**: Limite inferior da banda

```python
import pandas as pd

def calculate_bollinger_bands(df, window=20, num_std_dev=2):

    df = df.copy()  # Evita modificar o DataFrame original
    df['Bollinger_Middle'] = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    
    df['Bollinger_Upper'] = df['Bollinger_Middle'] + (rolling_std * num_std_dev)
    df['Bollinger_Lower'] = df['Bollinger_Middle'] - (rolling_std * num_std_dev)

    return df
```

## 9. Desvio padrão

O **desvio padrão** é uma medida estatística essencial que avalia o grau de dispersão dos valores em relação à média aritmética de um conjunto de dados. No contexto financeiro, ele é amplamente utilizado para medir a **volatilidade de um ativo**, ou seja, como os preços se distribuem ao longo do tempo em torno da média. Quando o desvio padrão é elevado, isso indica que os preços estão mais afastados da média, sinalizando maior volatilidade e, potencialmente, um cenário de maior risco para o investidor .

Essa métrica é fundamental na análise de risco e desempenho de ativos, pois permite avaliar a consistência ou a variabilidade dos retornos ao longo do tempo. Em outras palavras, um ativo com baixo desvio padrão tende a apresentar preços mais estáveis e previsíveis, enquanto um ativo com alto desvio padrão costuma ter oscilações mais intensas, tornando-se mais arriscado do ponto de vista de gestão de carteira .

Além disso, o desvio padrão desempenha um papel central no cálculo de diversos indicadores técnicos utilizados na análise gráfica, como as **Bandas de Bollinger**. Nesse caso, as bandas são construídas somando e subtraindo duas vezes o valor do desvio padrão à Média Móvel Simples (geralmente de 20 períodos), criando um canal dinâmico que se ajusta automaticamente conforme a volatilidade do mercado aumenta ou diminui .

Portanto, o desvio padrão não apenas ajuda a quantificar a volatilidade presente, mas também serve como base para ferramentas que auxiliam traders e analistas na identificação de oportunidades de negociação, como rompimentos, reversões e zonas de congestionamento .

$$
\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}
$$

- **σ**: Desvio padrão
- **N**: Número total de observações
- **xᵢ**: Valor de cada observação
- **μ**: Média dos valores

```python
import numpy as np

def desvio_padrao_manual(valores):
    n = len(valores)
    media = sum(valores) / n
    soma_quadrados = sum((x - media) ** 2 for x in valores)
    desvio = (soma_quadrados / n) ** 0.5
    return desvio
```

## 10. Retração de Fibonacci

Baseada na **sequência matemática de Fibonacci**, uma série na qual cada número é obtido pela soma dos dois termos imediatamente anteriores (como 0, 1, 1, 2, 3, 5, 8, 13, 21...), essa ferramenta técnica traça níveis percentuais-chave no gráfico de preços — tais como **38,2%, 50% e 61,8%** — que são amplamente utilizados para identificar possíveis zonas de **suporte ou resistência** durante correções dentro de uma tendência maior .

Esses níveis são derivados das relações matemáticas entre os números da sequência e estão associados à chamada **proporção áurea (~1,618)**, que se manifesta em diversos fenômenos naturais e artísticos. Na análise financeira, acreditasse que essas proporções podem indicar com maior probabilidade onde o preço pode encontrar reação ao buscar equilíbrio após uma movimentação inicial .

A aplicação mais comum ocorre em contextos de **retrações (pullbacks)** dentro de uma tendência estabelecida, permitindo ao trader identificar oportunidades de entrada alinhadas com a direção predominante do mercado. Por exemplo, em uma tendência de alta, um retrocesso que encontre suporte próximo ao nível de 38,2% ou 61,8% pode sinalizar uma continuidade da tendência principal, oferecendo pontos estratégicos para abertura de posições .

Além disso, os níveis de Fibonacci também são usados para projetar alvos de lucro e medir extensões do movimento de preço, ampliando sua utilidade além da simples identificação de zonas de reversão temporária .

Assim, a ferramenta de **retração de Fibonacci** é uma das mais populares na análise técnica por sua capacidade de fornecer referências objetivas em meio à subjetividade dos movimentos de mercado, sendo frequentemente combinada com outros indicadores para aumentar a precisão dos sinais gerados.

$$
F_n = F_{n-1} + F_{n-2}
$$

$$
Nível = H - \left( (H - L) \times \% \right)
$$

- *H* = Preço mais alto do movimento recente (High)
- *L* = Preço mais baixo do movimento recente (Low)
- % = Percentual de Fibonacci usado

```python
def fibonacci_retracements(high, low):
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low
    }
    return levels
```

## 11. Nuvem de Ichimoku

A **Nuvem de Ichimoku**, também conhecida como **Ichimoku Kinkō Hyō** (em japonês: 一目均衡表), é um sistema abrangente e multifacetado de análise técnica desenvolvido por Goichi Hosoda na década de 1930. Diferentemente de muitos outros indicadores, o Ichimoku oferece ao analista uma visão holística do mercado, integrando informações sobre **tendência**, **momento**, bem como **níveis potenciais de suporte e resistência** em um único gráfico .

O indicador é composto por **cinco linhas principais**, cada uma com uma fórmula matemática específica, e uma área visual chamada **"nuvem" (Kumo)**, que se forma entre duas dessas linhas. Essa nuvem atua como uma zona dinâmica de suporte ou resistência e fornece insights valiosos sobre a força e a direção da tendência.

### As cinco componentes do Ichimoku e suas fórmulas:

1. **Linha de Conversão (Tenkan-sen)**
    
    Representa a média entre a alta e a baixa de curto prazo (9 períodos) e indica mudanças no momentum de curta duração:
    
    $$
    \text{Tenkan-sen} = \frac{\text{Max. dos últimos 9 períodos + Min. dos últimos 9 períodos}}{2}
    $$
    
2. **Linha de Base (Kijun-sen)**
    
    Reflete a média entre a alta e a baixa de médio prazo (26 períodos), servindo como referência para tendências intermediárias:
    
    $$
    \text{Kijun-sen} = \frac{\text{Máxima dos últimos 26 períodos + Mínima dos últimos 26 períodos}}{2}
    $$
    
3. **Linha Leading Span A (Senkou Span A)**
    
    Representa a média móvel entre a Linha de Conversão e a Linha de Base, projetada 26 períodos à frente. Ela forma a borda superior ou inferior da nuvem:
    
    $$
    \text{Senkou Span A} = \frac{\text{Tenkan-sen + Kijun-sen}}{2} \quad \text{(projetada 26 períodos à frente)}
    $$
    
4. **Linha Leading Span B (Senkou Span B)**
    
    Calculada com base na média entre as máximas e mínimas de longo prazo (52 períodos), também projetada 26 períodos à frente, formando a outra borda da nuvem:
    
    $$
    \text{Senkou Span B} = \frac{\text{Máxima dos últimos 52 períodos + Mínima dos últimos 52 períodos}}{2} \quad \text{}
    $$
    
5. **Linha Lagging Span (Chikou Span)**
    
    Representa o preço de fechamento atual defasado em 26 períodos, usado para confirmar sinais com base em ações passadas do preço:
    
    $$
    \text{Chikou Span} = \text{Preço de fechamento atual (defasado 26 períodos)}
    $$
    

```python
import pandas as pd

def ichimoku_cloud(df):
   
    # Verificação de colunas
    if not {'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("DataFrame deve conter as colunas: 'High', 'Low', 'Close'")

    high = df['High']
    low = df['Low']
    close = df['Close']

    # Tenkan-sen (linha de conversão, 9 períodos)
    period9_high = high.rolling(window=9).max()
    period9_low = low.rolling(window=9).min()
    df['Tenkan_sen'] = (period9_high + period9_low) / 2

    # Kijun-sen (linha base, 26 períodos)
    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    df['Kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (média da Tenkan e Kijun, projetada 26 períodos à frente)
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    # Senkou Span B (média de 52 períodos, projetada 26 períodos à frente)
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    df['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (preço de fechamento, projetado 26 períodos para trás)
    df['Chikou_Span'] = close.shift(-26)

    return df
```

## 12. Volume de balanço (OBV)

O **On-Balance Volume (OBV)** , ou **Saldo de Volume** , é um indicador técnico de momentum que mede o fluxo acumulado de volume em um ativo ao longo do tempo, com base na variação do preço de fechamento. Ele opera sob uma lógica simples, porém poderosa: **quando o preço fecha acima da sessão anterior (alta), o volume daquele período é adicionado ao saldo acumulado** ; por outro lado, **se o preço fecha abaixo (queda), o volume é subtraído do saldo total.**

- **Se o preço de fechamento atual for maior que o preço de fechamento anterior**:

$$
\text{OBV}_t = \text{OBV}_{t-1} + V_t
$$

- **Se o preço de fechamento atual for menor que o preço de fechamento anterior**:

$$
\text{OBV}_t = \text{OBV}_{t-1} - V_t
$$

- **Se não houver mudança no preço (fechamento igual ao dia anterior)**:

$$
\text{OBV}_t = \text{OBV}_{t-1}

$$

- **OBVₜ**: On-Balance Volume no período atual
- **OBVₜ₋₁**: On-Balance Volume no período anterior
- **Vₜ**: Volume no período atual

```python
import pandas as pd
import numpy as np

def calcular_obv(df):
    
    obv = [0]  # Inicializa OBV com 0

    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])  # Sem variação no preço, OBV permanece o mesmo

    df['OBV'] = obv
    return df

```

## 13.Volume

O **volume** representa a quantidade total de unidades negociadas — sejam elas ações, contratos ou outros instrumentos financeiros — em um determinado período de tempo. Ele é uma métrica essencial na análise técnica e no entendimento do comportamento do mercado, pois reflete o **grau de participação e interesse dos traders e investidores** em relação a um ativo específico.

Quando há um **aumento significativo no volume**, especialmente durante movimentos de preço, isso pode indicar que há forte apoio por trás da tendência, seja para cima (alta) ou para baixo (baixa). Nesse contexto, o volume atua como um **validador do movimento**, reforçando a confiança nas decisões de compra ou venda . Por outro lado, **volumes reduzidos** durante uma movimentação de preço podem sugerir falta de convicção por parte dos participantes do mercado, possivelmente antecipando uma perda de força na tendência ou até mesmo uma reversão.

$$
\text{Volume} = \sum \left( \text{Qntd. de ativos negociados em cada transação no período} \right)
$$

```python
import pandas as pd
import numpy as np

def calcular_obv(df):
        obv = [0]  # Inicializa OBV com 0

    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])  # Sem variação no preço, OBV permanece o mesmo

    df['OBV'] = obv
    return df
```

## 14.Black-Scholes

O **modelo Black-Scholes**, também conhecido como **modelo Black-Scholes-Merton**, é uma das fórmulas matemáticas mais influentes e amplamente utilizadas na precificação de opções financeiras. Desenvolvido por Fischer Black, Myron Scholes e aprimorado por Robert Merton em 1973, o modelo introduziu uma abordagem rigorosa para calcular o **valor justo de uma opção europeia** com base em variáveis observáveis no mercado .

### Funcionamento do Modelo

O modelo se baseia na ideia de **replicação dinâmica**, ou seja, construir uma carteira que imita o retorno da opção utilizando ativos disponíveis no mercado, como o ativo subjacente e títulos livres de risco. Essa estratégia permite eliminar o risco específico da posição em opções, tornando possível precificá-la sem arbitragem .

A fórmula do **Black-Scholes para uma opção de compra (call)** é dada por:

$$
C(S,t)=S⋅N(d1​)−K⋅e−r(T−t)⋅N(d2​)
$$

E para uma opção de venda (put), pela paridade put-call:

$$
P(S,t)=K⋅e−r(T−t)⋅N(−d2​)−S⋅N(−d1​)
$$

Onde:

- C: preço da opção de compra
- P: preço da opção de venda
- S: preço atual do ativo subjacente
- K: preço de exercício da opção
- r: taxa de juros livre de risco
- T−t: tempo restante até o vencimento da opção
- σ: volatilidade do ativo subjacente
- N(⋅): função de distribuição acumulada da normal padrão

$$
d_1 = \frac{\ln\left(\frac{S}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)(T-t)}{\sigma \sqrt{T-t}}
\\d_2 = d_1 - \sigma \sqrt{T-t}
$$

### Principais Aplicações

O modelo Black-Scholes é amplamente utilizado por **investidores institucionais, traders e empresas financeiras** para precificar opções e gerenciar riscos associados às posições tomadas no mercado. Ele também serve como base para estratégias de hedge entre opções e seus ativos subjacentes . Além disso, o modelo ajuda os traders a identificar **opções subavaliadas ou superavaliadas** no mercado, oferecendo oportunidades de arbitragem .

### Suposições do Modelo

Apesar de sua robustez, o modelo se baseia em algumas suposições importantes, dentre elas:

- O mercado é eficiente e não apresenta oportunidades de arbitragem;
- A volatilidade do ativo subjacente é constante ao longo do tempo;
- Não há custos de transação ou impostos;
- As taxas de juros são constantes e conhecidas;
- O retorno do ativo subjacente segue uma distribuição log-normal.

Essas premissas, embora simplifiquem o cálculo, podem limitar a precisão do modelo em cenários reais de mercado, especialmente em períodos de alta volatilidade ou quando os preços desviam-se significativamente dos comportamentos esperados .

### Importância Histórica

A introdução do modelo Black-Scholes revolucionou o mercado financeiro, pois permitiu que os participantes do mercado controlassem melhor os riscos associados ao comércio de opções, contribuindo diretamente para o crescimento exponencial desse segmento. Isso levou à popularização de instrumentos derivativos e à criação de novos produtos financeiros baseados em opções .

Em reconhecimento à sua relevância, Myron Scholes e Robert Merton receberam o **Prêmio Nobel de Economia em 1997** (Fischer Black faleceu antes e, por isso, não foi agraciado) . Até hoje, o modelo permanece como um pilar fundamental da **matemática financeira** e da análise de risco no mercado de capitais.

```python
import math
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, tipo='call'):
   
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if tipo == 'call':
        preco = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif tipo == 'put':
        preco = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Tipo deve ser 'call' ou 'put'.")

    return preco
```