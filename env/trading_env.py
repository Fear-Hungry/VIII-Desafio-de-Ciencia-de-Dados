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
    ):
        super().__init__()

        # ---------- Polars DF ---------- #
        # Converta 'date' para pl.Date antes de instanciar o env!
        self.df = df.with_row_count()          # garante índice denso
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee
        self.price_col = price_col

        # Utiliza as features passadas como argumento
        self.features = features
        self.n_features = len(self.features)
        # Pré-computa as features selecionadas como um array NumPy
        self.features_np = self.df.select(self.features).to_numpy()

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
        # Acessa preço pela coluna configurada e índice, extraindo o escalar
        self.price_array = self.df[self.price_col].to_numpy()
        price = float(self.price_array[self.current_step])

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
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # use TimeLimit wrapper se quiser episódio finito

        # ---------- Recompensa Percentual ---------- #
        self.prev_net_worth = self.net_worth  # Guarda o valor anterior
        self.net_worth = self.balance + self.shares_held * price

        # Calcula reward percentual (evita divisão por zero ou por valor muito pequeno)
        if self.prev_net_worth > 1e-8:
            reward = (self.net_worth - self.prev_net_worth) / \
                self.prev_net_worth
        else:
            reward = 0.0  # Ou outra lógica, se net_worth puder ser 0 inicialmente

        # Opcional: Escalar reward com tanh para limitar entre -1 e 1
        # reward = np.tanh(reward * reward_scaling_factor) # reward_scaling_factor pode ser 1 ou outro valor

        info = {
            "balance": self.balance,
            "shares_held": self.shares_held,
            "net_worth": self.net_worth,
        }

        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):
        print(
            f"Step {self.current_step} | Balance {self.balance:,.2f} | "
            f"Shares {self.shares_held} | NetWorth {self.net_worth:,.2f}"
        )
