import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces


class CustomTradingEnv(gym.Env):
    """
    Ambiente de trading (0 = Hold, 1 = Buy, 2 = Sell) usando DataFrame Polars.
    Observação: janela 'window_size' × (n_features + 1), achatada em 1-D.
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        df: pl.DataFrame,
        features: list[str],
        window_size: int = 20,
        initial_balance: float = 1e5,
        fee: float = 0.001,
        price_col: str = "close",
        reward_scaling: float = 1.0,  # Fator para escala de recompensa
        episode_summary_freq: int = 1, # Frequência para calcular métricas de episódio (drawdown, sharpe)
        annualization_factor: float = 252.0 # Fator para anualizar Sharpe Ratio
    ):
        super().__init__()

        # ---------- Polars DF ---------- #
        # Converta 'date' para pl.Date antes de instanciar o env!
        self.df = df.with_row_count()          # garante índice denso
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee
        self.price_col = price_col
        self.reward_scaling = reward_scaling  # fator para tanh
        self.episode_summary_freq = episode_summary_freq
        self.annualization_factor = annualization_factor # Novo parâmetro

        # Utiliza as features passadas como argumento
        self.features = features
        self.n_features = len(self.features)
        # Pré-computa as features selecionadas como um array NumPy
        self.features_np = self.df.select(self.features).to_numpy()
        # Pré-computa o array de preços para acesso eficiente
        self.price_array_np = self.df[self.price_col].to_numpy().astype(np.float32)

        # ---------- VERIFICAÇÕES DE SANIDADE DOS DADOS ---------- #
        if np.any(np.isnan(self.features_np)):
            nan_counts = np.sum(np.isnan(self.features_np), axis=0)
            nan_cols_indices = np.where(nan_counts > 0)[0]
            nan_cols_names = [self.features[i] for i in nan_cols_indices]
            raise ValueError(
                f"NaNs encontrados em self.features_np nas colunas: {nan_cols_names}. "
                f"Contagens de NaN por coluna problemática: {nan_counts[nan_cols_indices]}"
            )
        if np.any(np.isinf(self.features_np)):
            inf_counts = np.sum(np.isinf(self.features_np), axis=0)
            inf_cols_indices = np.where(inf_counts > 0)[0]
            inf_cols_names = [self.features[i] for i in inf_cols_indices]
            raise ValueError(
                f"Infs encontrados em self.features_np nas colunas: {inf_cols_names}. "
                f"Contagens de Inf por coluna problemática: {inf_counts[inf_cols_indices]}"
            )

        if np.any(np.isnan(self.price_array_np)):
            raise ValueError(f"NaNs encontrados em self.price_array_np (coluna: {self.price_col})")
        if np.any(np.isinf(self.price_array_np)):
            raise ValueError(f"Infs encontrados em self.price_array_np (coluna: {self.price_col})")
        # ---------- FIM DAS VERIFICAÇÕES DE SANIDADE ---------- #

        # ---------- Spaces ---------- #
        self.action_space = spaces.Discrete(3)  # 0 hold, 1 buy, 2 sell
        obs_dim = self.window_size * (self.n_features + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_internal()

    # ------------------------------------------------------------------ #
    def _reset_internal(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.history_net_worth = [self.initial_balance] # Rastrear net_worth para drawdown/sharpe
        self.episode_steps = 0 # Contar passos no episódio

    # Gymnasium API ----------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._get_observation(), {}

    # ------------------------------------------------------------------ #
    def _get_observation(self):
        # Fatiar diretamente o array NumPy pré-computado
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        window = self.features_np[start_idx:end_idx]

        pos_col = np.full((self.window_size, 1),
                          self.shares_held, dtype=np.float32)
        obs = np.hstack((window, pos_col)).flatten().astype(np.float32)
        return obs

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        # Acessa preço pelo array pré-calculado
        price = self.price_array_np[self.current_step]

        # ---------- Executa ação ---------- #
        if action == 1 and self.balance >= price * (1 + self.fee):
            self.balance -= price * (1 + self.fee)
            self.shares_held += 1

        elif action == 2 and self.shares_held > 0:
            self.balance += price * (1 - self.fee)
            self.shares_held -= 1
        # else: hold

        # ---------- Avança ---------- #
        self.current_step += 1
        self.episode_steps += 1 # Incrementar contador de passos
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # use TimeLimit wrapper se quiser episódio finito

        # ---------- Recompensa Percentual ---------- #
        self.prev_net_worth = self.net_worth  # Guarda o valor anterior
        self.net_worth = self.balance + self.shares_held * price
        self.history_net_worth.append(self.net_worth) # Adicionar ao histórico

        # Calcula reward percentual (evita divisão por zero ou por valor muito pequeno)
        if self.prev_net_worth > 1e-8:
            reward = (self.net_worth - self.prev_net_worth) / \
                self.prev_net_worth
        else:
            reward = 0.0  # Ou outra lógica, se net_worth puder ser 0 inicialmente

        # Opcional: Escalar reward com tanh para limitar entre -1 e 1
        reward = np.tanh(reward * self.reward_scaling)  # normaliza recompensas

        info = {
            "balance": self.balance,
            "shares_held": self.shares_held,
            "net_worth": self.net_worth,
            "drawdown": 0.0,  # Valor padrão
            "sharpe": 0.0     # Valor padrão
        }

        if terminated and self.episode_steps > 0: # Calcular métricas no final do episódio
            drawdown, sharpe = self._calculate_episode_metrics()
            info["drawdown"] = drawdown
            info["sharpe"] = sharpe

        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _calculate_episode_metrics(self):
        """Calcula drawdown e Sharpe ratio para o episódio."""
        history = np.array(self.history_net_worth)
        if len(history) < 2:
            return 0.0, 0.0

        # Drawdown
        peak = np.maximum.accumulate(history)
        drawdown_series = (peak - history) / peak
        max_drawdown = np.max(drawdown_series)

        # Sharpe Ratio (simplificado, assumindo risk-free rate = 0 e retornos diários/por passo)
        returns = (history[1:] - history[:-1]) / history[:-1]
        # Substituir inf e -inf por NaN, depois por 0 para evitar problemas com std
        returns = np.where(np.isinf(returns), np.nan, returns)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        if np.std(returns) > 1e-8: # Evitar divisão por zero se não houver variação nos retornos
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(self.annualization_factor)
        else:
            sharpe_ratio = 0.0

        return max_drawdown, sharpe_ratio

    # ------------------------------------------------------------------ #
    def render(self):
        print(
            f"Step {self.current_step} | Balance {self.balance:,.2f} | "
            f"Shares {self.shares_held} | NetWorth {self.net_worth:,.2f}"
        )
