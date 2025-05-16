# %%
import polars as pl
import pandas as pd
from indicators.types import IndicatorConfig, IndicatorType
from indicators.momento import RSIIndicator, StochasticOscillatorIndicator, CCIIndicator
from indicators.volatilidade import ATRIndicator, BollingerBandsIndicator, DonchianChannelIndicator
from indicators.tendencia import IchimokuCloudIndicator
from indicators.volume import MFIIndicator, OBVIndicator, VWAPIndicator
from indicators.medias_moveis import ADXIndicator, EMAIndicator, SMAIndicator, MACDIndicator

from data_loader.loader import DataLoader

from backtesting import Backtest, Strategy
import optuna
from optuna.exceptions import TrialPruned

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import seaborn as sns
import matplotlib.pyplot as plt


# %%
df = pl.read_parquet("data/AAPL_5minute.parquet")
df.head()

# %%
def preprocess_with_indicators(path: str, indicadores: list[IndicatorConfig]) -> pl.DataFrame:
    # 1) Lazy load + limpezas básicas (usar with_columns para tudo)
    df = (
        pl.scan_parquet(path)  # Alterado de read_parquet para scan_parquet
          .sort("date")
          .fill_null(strategy="forward")
          .fill_null(0.0)
          .with_columns([
              pl.col(c).log().alias(f"{c}_log")
              for c in ["open", "high", "low", "close"]
          ])
          .collect()
    )

    # 2) Adiciona indicadores via DataLoader
    dl = DataLoader()
    df = dl.add_technical_indicators(df, indicators_to_add=indicadores)

    # 3) Reaplica fill_null para cobrir NaNs iniciais dos indicadores
    df = df.fill_null(strategy="forward").fill_null(0.0)

    # 4) Padroniza nomes de colunas para lowercase para evitar case mismatches
    df = df.rename({col: col.lower() for col in df.columns})

    return df

# %% [markdown]
# Pre-processamento

# %%
indicadores = [
    # Momento
    IndicatorConfig(IndicatorType.RSI, params=[14]),
    IndicatorConfig(IndicatorType.STOCH, params=[14, 3]), # k_period, d_period, slowing_k_period
    IndicatorConfig(IndicatorType.CCI, params=[20]),

    # Volatilidade
    IndicatorConfig(IndicatorType.BB, params=[20, 2]), # period, num_std_devs
    IndicatorConfig(IndicatorType.DONCHIAN, params=[20]), # n_periods
    # IndicatorConfig(IndicatorType.ROC, params=[12]), # Você removeu ROCIndicator da importação

    # Tendência
    IndicatorConfig(IndicatorType.ICHIMOKU, params=[9, 26, 52]), # tenkan_sen_period, kijun_sen_period, senkou_span_b_period

    # Volume
    IndicatorConfig(IndicatorType.MFI, params=[14]), # n_periods
    IndicatorConfig(IndicatorType.OBV, params=[]),
    IndicatorConfig(IndicatorType.VWAP, params=[14]),

    # Médias Móveis
    IndicatorConfig(IndicatorType.ADX, params=[14]), # n_periods
    IndicatorConfig(IndicatorType.EMA, params=[20]), # n_periods
    IndicatorConfig(IndicatorType.SMA, params=[50]), # n_periods
    IndicatorConfig(IndicatorType.MACD, params=[12, 26, 9]) # fast_period, slow_period, signal_period
]

# %%
path = "/workspaces/VIII-Desafio-de-Ciencia-de-Dados/data/AAPL_5minute.parquet"
df = preprocess_with_indicators(path, indicadores)

# %%
df

# %%
print(df.shape)
print(df.columns)


# %%
df.select([pl.count().alias("nulos")]).to_pandas()
df.describe().to_pandas()


# %%
import matplotlib.pyplot as plt
# Convertendo para pandas antes de plotar
pdf_rsi = df.select(['date', 'rsi_14']).to_pandas()
plt.plot(pdf_rsi["date"], pdf_rsi["rsi_14"])
plt.title("RSI (14) ao longo do tempo")
plt.show()


# %%
# Convertendo para pandas antes de plotar
pdf_rsi = df["rsi_14"].to_pandas()
plt.hist(pdf_rsi, bins=50)
plt.axvline(30, color="r", linestyle="--")
plt.axvline(70, color="r", linestyle="--")
plt.title("Distribuição do RSI (14)")
plt.show()


# %% [markdown]
# O histograma mostra um RSI bem "gaussiano" em torno dos 50, com quase toda a massa entre 30 e 70 — ou seja, sinais de sobrecompra/sobrevenda (<30/>70) são relativamente raros.
#

# %%
total = len(df)
oversold = (df["rsi_14"] < 30).sum()
overbought = (df["rsi_14"] > 70).sum()

print(f"Oversold (<30): {oversold} barras — {oversold/total:.2%}")
print(f"Overbought (>70): {overbought} barras — {overbought/total:.2%}")


# %% [markdown]
# Tentando suavizar o RSI, transformando para 5 minutuos

# %%
df = df.with_columns([
    pl.col("rsi_14")
      .rolling_mean(window_size=5)
      .alias("RSI_smooth")
])


# %%
total = len(df)
os_s = (df["RSI_smooth"] < 30).sum()
ob_s = (df["RSI_smooth"] > 70).sum()
print(f"Oversold_smooth: {os_s} barras — {os_s/total:.2%}")
print(f"Overbought_smooth: {ob_s} barras — {ob_s/total:.2%}")


# %%
import matplotlib.pyplot as plt

# converte para pandas (se ainda não for)
rsi_s = df["RSI_smooth"].to_pandas()

plt.figure(figsize=(8,4))
plt.hist(rsi_s, bins=50, edgecolor="k")
plt.axvline(30, color="r", linestyle="--", label="30 / oversold")
plt.axvline(70, color="r", linestyle="--", label="70 / overbought")
plt.title("Distribuição do RSI suavizado (5-period SMA)")
plt.legend()
plt.show()


# %%

# --- Função de backtest para indicadores --- #
def backtest_indicators(df, indicators):
    """
    Função para backtest de estratégias baseadas em indicadores técnicos.

    Args:
        df: DataFrame (pandas ou polars) com dados OHLCV
        indicators: dicionário de indicadores com thresholds de compra/venda

    Returns:
        pandas DataFrame com resultados do backtest
    """
    results = []

    # Garantir que estamos trabalhando com pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Convertendo Polars DataFrame para pandas DataFrame...")
        df = df.to_pandas()

    # Renomear colunas para padrão do Backtest
    df_bt = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}) \
              .dropna(subset=['Open','High','Low','Close'])

    class IndicatorStrategy(Strategy):
        def init(self):
            self.ind = self.I(lambda d: d[self.name], self.data.df)
        def next(self):
            if not self.position and self.ind[-1] < self.buy_thr:
                self.buy()
            elif self.position and self.ind[-1] > self.sell_thr:
                self.position.close()

    for name, thr in indicators.items():
        Strat = type(f"Strat_{name}", (IndicatorStrategy,),
                     {"name": name, "buy_thr": thr['buy_thr'], "sell_thr": thr['sell_thr']})
        bt = Backtest(df_bt, Strat, cash=100_000, commission=0.001, trade_on_close=True)
        stats = bt.run()
        bt.plot()  # Adicionado para plotar o gráfico da estratégia
        results.append({
            'indicator': name,
            'Return [%]': stats['Return [%]'],
            'Sharpe': stats['Sharpe Ratio']
        })
    return pd.DataFrame(results).set_index('indicator')


# %% [markdown]
# # Análise de Correlação entre Indicadores

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1) Lista de colunas de indicador, como antes
exclude = {
    'open','high','low','close','volume','date',
    'open_log','high_log','low_log','close_log'
}
ind_cols = [c for c in df.columns if c not in exclude]

# --- 2) Converta só o pedaço de indicadores para pandas
pdf_ind = df[ind_cols].to_pandas()

# --- 3) Calcule correlação absoluta
corr = pdf_ind.corr().abs()

# --- 4) Plote o heatmap
plt.figure(figsize=(max(10, corr.shape[1]*0.3), max(8, corr.shape[0]*0.3)))
plt.imshow(corr, vmin=0, vmax=1, aspect='auto')
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.index)
plt.title("Matriz de Correlação Absoluta dos Indicadores")
plt.tight_layout()
plt.show()

# --- 5) Extraia pares com |corr| > thresh
thresh = 0.95
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
high_corr = (
    corr.where(mask)         # zera abaixo da diagonal
        .stack()             # empilha (i,j) e valor
        .reset_index(name='corr')
        .rename(columns={'level_0':'feat1','level_1':'feat2'})
)
high_corr = high_corr[high_corr['corr'] > thresh] \
            .sort_values('corr', ascending=False)

print(f"\n{len(high_corr)} pares com |corr| > {thresh}:\n", high_corr)

# --- 6) Drop automático das segundas colunas de cada par
# supondo que `to_drop` é um set ou lista de strings
cols_to_drop = high_corr['feat2'].unique().tolist()  # pega cada indicador redundante apenas uma vez

# abordagem 1: usando Polars .drop()
df_reduzido = df.drop(cols_to_drop)

# OU, se preferir via select:
# df_reduzido = df.select([c for c in df.columns if c not in cols_to_drop])

print("Colunas removidas:", cols_to_drop)
print("Forma original:", df.shape)
print("Forma reduzida:", df_reduzido.shape)

# %%
# Primeiro, vamos imprimir os nomes exatos das colunas de indicadores para referência
print("Nomes exatos das colunas de indicadores:")
indicator_columns = [col for col in df.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume',
                                                         'open_log', 'high_log', 'low_log', 'close_log']]
print(indicator_columns)

# %%
# Exemplo de comparação – usando nomes de colunas exatos em minúsculas
indicator_thresholds = {
    'rsi_14':    {'buy_thr': 30,  'sell_thr': 70},
    'cci_20':    {'buy_thr': -100,'sell_thr': 100},
    # Use os nomes exatos das colunas aqui, conforme o print acima
}

# 1) Converte todo o Polars DataFrame original pra pandas
pdf_full = df.to_pandas()
pdf_full['date'] = pd.to_datetime(pdf_full['date'])
pdf_full.set_index('date', inplace=True)

# 2) Converte também o df_reduzido
pdf_reduced = df_reduzido.to_pandas()
pdf_reduced['date'] = pd.to_datetime(pdf_reduced['date'])
pdf_reduced.set_index('date', inplace=True)

# 3) Agora chame backtest_indicators usando esses pandas DataFrames
summary_full    = backtest_indicators(pdf_full,    indicator_thresholds)
summary_reduced = backtest_indicators(pdf_reduced, indicator_thresholds)

print("=== Backtest Completo ===")
print(summary_full)
print("\n=== Backtest Reduzido ===")
print(summary_reduced)


# %% [markdown]
# # Treinamento DRL

# %%
df_reduzido.columns

# %%
df = df_reduzido.to_pandas()
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df

# %%
feature_names = [
    'close', 'volume',
    'rsi_14',
    'stochk_14_3',
    'stochd_14_3',
    'cci_20',
    'bb_middle_20',
    'mfi_14',
    'adx_14',
    '+di_14',
    '-di_14',
    'macd_line',
    'macd_hist',
]

# %%
from env.trading_env import CustomTradingEnv

def make_env(df):
    def _init():
        return CustomTradingEnv(df, features=feature_names, window_size=20)
    return _init


# %%
print(f"Colunas em df_reduzido para DRL: {df_reduzido.columns}")

train_ratio = 0.8
n_total_rows = df_reduzido.height # Usar .height para Polars
n_train_rows = int(n_total_rows * train_ratio)

# Divide o DataFrame Polars
df_train_pl = df_reduzido.slice(0, n_train_rows)
df_test_pl = df_reduzido.slice(n_train_rows, n_total_rows - n_train_rows) # O segundo argumento é o tamanho

print(f"Total de linhas no df_reduzido: {n_total_rows}")
print(f"Linhas de treino (Polars): {df_train_pl.height}")
print(f"Linhas de teste (Polars): {df_test_pl.height}")

# %%
train_env_initializer = make_env(df_train_pl)
test_env_initializer = make_env(df_test_pl)

print("Inicializadores de ambiente de treino e teste para DRL foram criados.")

# %%
pdf = df_reduzido.to_pandas()
pdf['date'] = pd.to_datetime(pdf['date'])
pdf.set_index('date', inplace=True)

split_idx = int(len(pdf) * 0.8)
train_df_pd = pdf.iloc[:split_idx]
test_df_pd  = pdf.iloc[split_idx:]

# Adiciona colunas uppercase necessárias para o backtesting.py (mantém lowercase para RL)
test_df_pd['Open']  = test_df_pd['open']
test_df_pd['High']  = test_df_pd['high']
test_df_pd['Low']   = test_df_pd['low']
test_df_pd['Close'] = test_df_pd['close']

# %%
def make_env_optuna(df_input: pd.DataFrame, window_size: int = 20, fee: float = 0.001):

    df_polars = pl.from_pandas(df_input.reset_index()) # Converte para Polars

    def _init():
        env = CustomTradingEnv(df_polars, features=feature_names, window_size=window_size, fee=fee)
        return TimeLimit(env, max_episode_steps=df_polars.height - window_size)

    return _init

# %%
"""class TensorboardCallback(BaseCallback):
    def __init__(self, eval_freq: int, eval_env=None, n_eval_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq  # Armazena eval_freq
        # Se esta callback for realizar sua própria avaliação para logar,
        # você também precisará do eval_env e n_eval_episodes.
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0: # Usa self.eval_freq
            if self.eval_env is not None:
                # Se esta callback deve fazer sua própria avaliação:
                # mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True)
                # self.logger.record("custom_eval/mean_reward", mean_reward)
                # print(f"TensorboardCallback: Eval @ step {self.n_calls}, Mean reward: {mean_reward}")
                pass # Remova o 'pass' e descomente/implemente a lógica de avaliação e log acima se necessário.
            else:
                # Se esta callback loga outras coisas ou você espera que 'mean_reward' venha de outro lugar.
                # A linha 'mean_reward = ... # cálculo' no seu traceback indica que o cálculo está pendente.
                # Por agora, vamos apenas resolver o NameError.
                if self.verbose > 0:
                    print(f"TensorboardCallback: _on_step called at {self.n_calls} (eval_freq: {self.eval_freq}). Implement logging logic.")
        return True"""

# %%
class OptunaCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, eval_env: VecNormalize, n_eval_episodes: int = 5, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Usar return_episode_rewards=True para obter a lista de recompensas
            episode_rewards, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True
                # O warning sobre Monitor não deve aparecer aqui se eval_env já tem Monitor
            )
            # Calcular média e desvio padrão manualmente
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            if self.verbose > 0:
                print(f"Eval @ step {self.n_calls}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            # Reporta a métrica para o Optuna
            self.trial.report(mean_reward, self.n_calls)

            # Pruning (interrompe o trial se não estiver melhorando)
            if self.trial.should_prune():
                if self.verbose > 0:
                    print("Trial pruned by Optuna.")
                raise TrialPruned()

            # Opcional: salvar o melhor modelo intermediário do trial
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # self.model.save(f"trial_{self.trial.number}_best_model_step_{self.n_calls}")
        return True

# %%
import os

def objective(trial: optuna.Trial) -> float:
    # --- Hiperparâmetros a buscar ---
    n_steps         = trial.suggest_categorical('n_steps', [128, 256, 512, 1024])
    gamma           = trial.suggest_float('gamma', 0.90, 0.9999, log=True)
    learning_rate   = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef        = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True) # Aumentei um pouco o limite superior
    batch_size      = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Novos hiperparâmetros sugeridos
    vf_coef         = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm   = trial.suggest_float("max_grad_norm", 0.3, 5.0)
    clip_range      = trial.suggest_float("clip_range", 0.1, 0.4)
    norm_reward_train = trial.suggest_categorical('norm_reward_train', [True, False])
    gae_lambda      = trial.suggest_float("gae_lambda", 0.9, 0.999)

    # Sugerir arquitetura de rede
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium"])
    if net_arch_choice == "small":
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    else: # medium
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

     # --- Criação de diretórios de log para o Monitor ---
    log_dir_base = "./monitor_logs_optuna" # Diretório base para todos os logs de monitoramento
    trial_log_dir = os.path.join(log_dir_base, f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    train_monitor_path = os.path.join(trial_log_dir, "train") # Monitor adiciona .monitor.csv
    eval_monitor_path = os.path.join(trial_log_dir, "eval")   # Monitor adiciona .monitor.csv


    # --- Cria ambientes vetorizados ---
    # Ambiente de Treino
    # Assegure que make_env_optuna retorna uma função lambda: YourEnv(df)
    train_env_callable = make_env_optuna(train_df_pd)
    train_env_monitored_callable = lambda: Monitor(train_env_callable(), filename=train_monitor_path, allow_early_resets=True)
    train_env_raw = DummyVecEnv([train_env_monitored_callable])

    # Envolver com Monitor ANTES de VecNormalize se quiser logar recompensas não normalizadas
    # train_env_monitored = DummyVecEnv([lambda: Monitor(train_env_callable())])
    train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=norm_reward_train, training=True, gamma=gamma)


    # Ambiente de Avaliação
    eval_env_callable = make_env_optuna(test_df_pd)
    # Envolver com Monitor para avaliação correta
    eval_env_monitored_callable = lambda: Monitor(eval_env_callable(), filename=eval_monitor_path, allow_early_resets=True)
    eval_env_raw = DummyVecEnv([eval_env_monitored_callable])
    eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, training=False, gamma=gamma)
    # Sincronizar estatísticas do train_env para eval_env ANTES da avaliação e do callback
    eval_env.obs_rms = train_env.obs_rms # Copia obs_rms
    # eval_env.ret_rms = train_env.ret_rms # Não copie ret_rms se norm_reward_train=True e eval_env.norm_reward=False

    # --- Callback para Optuna Pruning ---
    # Aumentando n_eval_episodes e ajustando eval_freq
    # total_timesteps_learn é o total de passos para model.learn
    total_timesteps_learn = 50_000
    eval_freq_callback = max(1, total_timesteps_learn // 20) # Avalia ~10 vezes durante o treino
    n_eval_episodes_callback = 10

    optuna_callback = OptunaCallback(
        trial,
        eval_env, # Passa o ambiente de avaliação normalizado
        n_eval_episodes=n_eval_episodes_callback,
        eval_freq=eval_freq_callback, # Frequência de avaliação dentro do learn
        verbose=0 # Defina para 1 para ver logs do callback
    )

    # --- Instância e treino do PPO ---
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tb_logs/",
        verbose=1
    )

    try:
        model.learn(total_timesteps=total_timesteps_learn, callback=[optuna_callback])
    except TrialPruned:
        # Se o trial foi interrompido pelo callback, propaga a exceção
        # para que o Optuna saiba que foi podado.
        train_env.close()
        eval_env.close()
        raise
    except Exception as e:
        # Em caso de outros erros, feche os ambientes e levante a exceção
        print(f"An error occurred during model training: {e}")
        train_env.close()
        eval_env.close()
        raise

    # --- Avalia política no ambiente de teste após o treino completo ---
    # Sincronizar estatísticas do train_env para eval_env novamente caso tenham mudado
    eval_env.obs_rms = train_env.obs_rms
    # eval_env.ret_rms = train_env.ret_rms # Cuidado aqui

    n_eval_episodes_final = 10
    # Usar return_episode_rewards=True para obter a lista de recompensas
    episode_rewards, _ = evaluate_policy(
        model,
        eval_env, # Usar o ambiente de avaliação normalizado e com Monitor
        n_eval_episodes=n_eval_episodes_final,
        deterministic=True,
        return_episode_rewards=True
    )
    # Calcular média e desvio padrão manualmente
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # Log de estatísticas (opcional)
    if hasattr(train_env, 'obs_rms') and train_env.obs_rms is not None:
        trial.set_user_attr("train_obs_rms_mean", float(train_env.obs_rms.mean.mean()))
        trial.set_user_attr("train_obs_rms_var_mean", float(train_env.obs_rms.var.mean()))

    # Opcionalmente, adicionar o desvio padrão como user attribute
    trial.set_user_attr("eval_reward_std", float(std_reward))

    train_env.close() # Fechar ambientes
    eval_env.close()

    # Optuna vai maximizar esse valor
    # Se quiser usar avg - k*std: return mean_reward - (k * std_reward)
    return mean_reward

# %%
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
# study.optimize retorna None, então não devemos atribuir seu retorno a uma variável
study.optimize(objective, n_trials=100)

print("✔️ Melhor conjunto de parâmetros:")
print(study.best_params)
print(f"✔️ Recompensa média obtida: {study.best_value:.2f}")

# %%

best_params = study.best_params

final_train_env_raw = DummyVecEnv([make_env_optuna(train_df_pd, window_size=best_params.get('window_size', 20))])
final_train_env = VecNormalize(final_train_env_raw, norm_obs=True, norm_reward=True)

model = PPO(
    "MlpPolicy",
    final_train_env,
    n_steps=best_params['n_steps'],
    gamma=best_params['gamma'],
    learning_rate=best_params['learning_rate'],
    ent_coef=best_params['ent_coef'],
    batch_size=best_params['batch_size'],
    verbose=1
)
model.learn(total_timesteps=50_000)


model.save("models/best/ppo_trading_best")
final_train_env.save("models/best/vecnormalize_trading_best.pkl")

print("🎉 Treinamento finalizado e modelo salvo.")


# %%
# Não existe valor em 'reward' já que study.optimize() retorna None
# Substituindo por uma informação útil
print(f"Melhor valor encontrado: {study.best_value}")

# %%
PATH_MODELO_PPO = "./models/best/ppo_trading_best.zip"
PATH_VECNORMALIZE = "./models/best/vecnormalize_trading_best.pkl"

try:
    modelo_carregado = PPO.load(PATH_MODELO_PPO)
    print(f"Modelo PPO carregado de {PATH_MODELO_PPO}")
except Exception as e:
    print(f"Erro ao carregar o modelo PPO de {PATH_MODELO_PPO}: {e}")
    modelo_carregado = None

# %%
try:
    env_normalizado_stats = VecNormalize.load(PATH_VECNORMALIZE, DummyVecEnv([make_env_optuna(test_df_pd)]))
    print(f"Estatísticas do VecNormalize carregadas de {PATH_VECNORMALIZE}")

    backtest_env_raw = DummyVecEnv([make_env_optuna(test_df_pd)])
    backtest_env = VecNormalize(backtest_env_raw, training=False, norm_obs=True, norm_reward=False)
    backtest_env.obs_rms = env_normalizado_stats.obs_rms
    backtest_env.clip_obs = env_normalizado_stats.clip_obs
    backtest_env.epsilon = env_normalizado_stats.epsilon
    print("Ambiente de backtesting VecNormalize preparado.")
except Exception as e:
    print(f"Erro ao carregar ou preparar o VecNormalize de {PATH_VECNORMALIZE}: {e}")
    print("Continuando sem normalização de observação específica do VecNormalize salvo (isto pode afetar o desempenho).")
    backtest_env_raw = DummyVecEnv([make_env_optuna(test_df_pd)])
    backtest_env = VecNormalize(backtest_env_raw, training=False, norm_obs=True, norm_reward=False)
    print("Ambiente de backtesting VecNormalize criado com novas estatísticas (pode não ser o ideal).")


# %%
if modelo_carregado and backtest_env:
    # --- 2. Preparar para Coletar Dados ---
    obs = backtest_env.reset()
    done = False

    portfolio_values = []
    buy_signals_idx = []
    sell_signals_idx = []
    actual_rewards = [] # Recompensas reais do ambiente, não normalizadas

    # Supondo que seu test_df_pd tem uma coluna de preço, ex: 'Close'
    # para plotar junto com os sinais
    precos_teste = test_df_pd['Close'].values[-len(test_df_pd) + modelo_carregado.n_steps -1:] # Ajuste o índice conforme necessário

    current_step_in_episode = 0

    # --- 3. Executar o Agente e Coletar Dados ---
    # Execute por um número de passos ou até o final do dataset de teste
    num_backtest_steps = len(test_df_pd) - 1 # Ou um valor menor para um backtest mais curto

    print(f"Iniciando backtesting por {num_backtest_steps} passos...")

    for i in range(num_backtest_steps):
        action, _states = modelo_carregado.predict(obs, deterministic=True)
        obs, reward, done, info_list = backtest_env.step(action)
        info = info_list[0] # DummyVecEnv retorna uma lista de infos

        actual_rewards.append(reward[0]) # reward também é uma lista/array com DummyVecEnv

        # Supondo que seu 'info' contenha esses dados:
        if 'portfolio_value' in info:
            portfolio_values.append(info['portfolio_value'])

        # Exemplo: Supondo que a ação 1 é COMPRAR e a ação 2 é VENDER
        # E que seu ambiente retorna a ação efetivamente tomada ou um sinal
        # Adapte isso à sua definição de espaço de ação e ao que 'info' retorna
        if 'trade_type' in info:
            if info['trade_type'] == 'BUY':
                buy_signals_idx.append(i) # Salva o índice do passo (ou data)
            elif info['trade_type'] == 'SELL':
                sell_signals_idx.append(i)
        elif 'action_taken' in info: # Alternativa se info tem a ação
             if info['action_taken'] == 1: # Exemplo para COMPRAR
                 buy_signals_idx.append(i)
             elif info['action_taken'] == 2: # Exemplo para VENDER
                 sell_signals_idx.append(i)


        if done:
            print(f"Episódio finalizado no passo {i}.")
            # Se você quiser continuar por múltiplos episódios no dataset de teste:
            # obs = backtest_env.reset()
            # current_step_in_episode = 0
            # Ou pare se for um backtest de um único "passe":
            break

        current_step_in_episode += 1
        if (i + 1) % 1000 == 0:
            print(f"Backtesting: passo {i+1}/{num_backtest_steps}")

    backtest_env.close()
    print("Backtesting finalizado.")

    # --- 4. Visualizar os Resultados ---
    print("Gerando visualizações...")
    num_collected_points = len(portfolio_values) if portfolio_values else len(actual_rewards)

    if num_collected_points > 0:
        precos_plot = test_df_pd['Close'].iloc[:num_collected_points].values

        plt.figure(figsize=(15, 10))

        # Plot do Valor da Carteira
        if portfolio_values:
            plt.subplot(2, 1, 1)
            plt.plot(portfolio_values, label='Valor da Carteira')
            plt.title('Evolução do Valor da Carteira Durante o Backtesting')
            plt.xlabel('Passos de Tempo')
            plt.ylabel('Valor da Carteira')
            plt.legend()
            plt.grid(True)

        # Plot de Preços com Sinais de Compra/Venda
        # (Certifique-se de que precos_plot tem o mesmo comprimento que o período de backtest)
        plt.subplot(2, 1, 2)
        plt.plot(precos_plot, label='Preço do Ativo (Close)', alpha=0.7)

        # Plotar os sinais. É importante que os índices em buy_signals_idx e sell_signals_idx
        # correspondam aos índices do array precos_plot.
        if buy_signals_idx:
            plt.scatter([idx for idx in buy_signals_idx if idx < len(precos_plot)],
                        precos_plot[[idx for idx in buy_signals_idx if idx < len(precos_plot)]],
                        label='Compra', marker='^', color='green', s=100, alpha=1)
        if sell_signals_idx:
            plt.scatter([idx for idx in sell_signals_idx if idx < len(precos_plot)],
                        precos_plot[[idx for idx in sell_signals_idx if idx < len(precos_plot)]],
                        label='Venda', marker='v', color='red', s=100, alpha=1)

        plt.title('Preço do Ativo com Sinais de Compra/Venda')
        plt.xlabel('Passos de Tempo')
        plt.ylabel('Preço')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot de Recompensas Acumuladas
        if actual_rewards:
            plt.figure(figsize=(15, 5))
            cumulative_rewards = pd.Series(actual_rewards).cumsum()
            plt.plot(cumulative_rewards, label='Recompensa Acumulada (Backtest)')
            plt.title('Recompensa Acumulada Durante o Backtesting')
            plt.xlabel('Passos de Tempo')
            plt.ylabel('Recompensa Acumulada')
            plt.legend()
            plt.grid(True)
            plt.show()
            print(f"Recompensa total acumulada no backtest: {cumulative_rewards.iloc[-1]}")

    else:
        print("Nenhum dado foi coletado para plotagem.")
else:
    print("Não foi possível carregar o modelo ou preparar o ambiente para o backtesting.")

# %%
# --- Backtesting com backtesting.py usando RLStrategy ---
window_size = 20

class RLStrategy(Strategy):
    def init(self):
        # Carrega o modelo e o normalizador
        self.model = PPO.load("models/best/ppo_trading_best.zip")
        # Crio um DummyVecEnv válido para carregar estatísticas do VecNormalize
        # Converto o DataFrame do backtesting (pandas) para Polars
        df_polars = pl.from_pandas(self.data.df.reset_index())
        dummy_env = DummyVecEnv([lambda: CustomTradingEnv(
            df_polars,
            features=feature_names,
            window_size=window_size)])

        self.vecnorm = VecNormalize.load(
            "models/best/vecnormalize_trading_best.pkl",
            dummy_env)
        self.model.set_env(self.vecnorm)

    def next(self):
        # Extrai janela de features e faz predição
        df_window = self.data.df[feature_names]
        obs = df_window.iloc[self.now - window_size + 1 : self.now + 1].values
        action, _ = self.model.predict(obs, deterministic=True)
        if action == 1 and not self.position:
            self.buy()
        elif action == 2 and self.position:
            self.position.close()

# Prepara o DataFrame de teste para o backtest
pdf_test = test_df_pd.copy()
pdf_test.index.name = "date"

# Executa o backtest com a estratégia RL
bt = Backtest(
    pdf_test,
    RLStrategy,
    cash=100_000,
    commission=0.001,
    trade_on_close=True
)
stats = bt.run()
bt.plot()
print(stats)

# Defino cumret a partir da curva de equity do backtest
cumret = bt._equity_curve['Equity']

pdf_test = df

# %%
cum = cumret
running_max = cum.cummax()
drawdown = (cum - running_max) / running_max

plt.figure(figsize=(10,4))
plt.plot(drawdown, color='tab:red')
plt.title("Drawdown")
plt.axhline(drawdown.min(), color='black', linestyle='--',
            label=f"Max Drawdown {drawdown.min():.2%}")
plt.legend()
plt.show()

# %%
# selecione só as colunas de indicadores
ind_cols = [c for c in df.columns if c not in ['date','open','high','low','close','volume']]
pdf = df[ind_cols]

# histograma e boxplot lado a lado
fig, axes = plt.subplots(len(ind_cols),2, figsize=(12,4*len(ind_cols)))
for i, col in enumerate(ind_cols):
    sns.histplot(pdf[col], bins=50, ax=axes[i,0], kde=True)
    axes[i,0].set_title(f'Histograma {col}')
    sns.boxplot(x=pdf[col], ax=axes[i,1])
    axes[i,1].set_title(f'Boxplot {col}')
plt.tight_layout()
plt.show()

# %%
