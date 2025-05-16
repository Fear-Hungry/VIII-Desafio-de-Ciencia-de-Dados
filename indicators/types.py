from enum import Enum
from typing import List, Optional, Union

import logging
logger = logging.getLogger(__name__)

# Configuração base para todos os indicadores
#@dataclass

class IndicatorType(Enum):
    """Enumeração dos tipos base de indicadores técnicos."""

    SMA = "sma"
    EMA = "ema"
    MACD = "macd"
    RSI = "rsi"
    BB = "bb"  # Bollinger Bands
    ADX = "adx"  # Average Directional Index
    ROC = "roc"  # Rate of Change
    STOCH = "stoch"  # Stochastic Oscillator
    FIBONACCI = "fibonacci"  # Fibonacci Retracement
    ICHIMOKU = "ichimoku"  # Ichimoku Cloud
    CCI = "cci"  # Commodity Channel Index
    VWAP = "vwap"  # Volume Weighted Average Price
    MFI = "mfi"  # Money Flow Index
    DONCHIAN = "donchian"  # Donchian Channel
    OBV = "obv"  # On-Balance Volume
    # Fibonacci (para compatibilidade com o nome usado no notebook)
    FIBO = "fibo"

    def __str__(self):
        return self.value


class IndicatorConfig:
    """Configuração para uma instância específica de um indicador técnico."""

    def __init__(
        self, type: IndicatorType, params: Optional[List[Union[int, float]]] = None
    ):
        self.type = type
        self.params = (
            params if params is not None else self._get_default_params(type, params)
        )
        self._validate_params()
        self._column_name = self._generate_column_name()

    def _get_default_params(
        self, type: IndicatorType, params: Optional[List[Union[int, float]]] = None
    ) -> List[Union[int, float]]:
        """Retorna os parâmetros padrão para tipos de indicadores conhecidos."""
        defaults = {
            IndicatorType.MACD: [12, 26, 9],
            IndicatorType.RSI: [14],
            IndicatorType.BB: [20, 2.0],
            IndicatorType.ADX: [14],
            IndicatorType.ROC: [14],
            IndicatorType.STOCH: [14, 3],  # Padrão k=14, d=3
        }
        # Retorna o padrão se existir
        if type in defaults:
            return defaults[type]
        # Se não tem padrão e não foram dados parâmetros, erro (exceto SMA/EMA que são tratados na validação)
        if params is None and type not in [IndicatorType.SMA, IndicatorType.EMA]:
            if type == IndicatorType.ICHIMOKU:
                raise ValueError("Ichimoku Cloud requer 3 parâmetros")
            if type == IndicatorType.FIBONACCI:
                raise ValueError(
                    "O parâmetro 'period' para Fibonacci é obrigatório e deve ser um inteiro positivo."
                )
            raise ValueError(
                f"Parâmetros são obrigatórios para o tipo de indicador '{type.name}' e nenhum padrão está definido."
            )
        # Se params foi dado mas não há padrão, retorna os params (validação pegará erro se faltar para SMA/EMA)
        if params is not None:
            return params
        # Caso contrário (SMA/EMA sem params), a validação dará erro
        # Para evitar erro de tipo aqui, retornamos lista vazia, validação falhará
        return []

    def _validate_params(self):
        """Valida os parâmetros com base no tipo de indicador."""
        if not hasattr(self, "params") or self.params is None:
            # Se chegou aqui sem params e não é um tipo com default, algo está errado
            # A validação de SMA/EMA trata o caso de params obrigatórios
            if self.type not in [IndicatorType.SMA, IndicatorType.EMA]:
                # Tipos que *precisam* de params mas não têm default (SMA/EMA)
                raise ValueError(
                    f"Parâmetros não foram fornecidos ou inicializados para {self.type.name}"
                )
            # Para outros tipos, se params é None, _get_default_params já deu erro ou retornou default

        if not isinstance(self.params, list):
            raise TypeError(
                f"Parâmetros para {self.type.name} devem ser uma lista, recebido: {self.params}"
            )

        # Validações específicas por tipo
        if self.type in [
            IndicatorType.SMA,
            IndicatorType.EMA,
            IndicatorType.RSI,
            IndicatorType.ROC,
        ]:
            if (
                len(self.params) != 1
                or not isinstance(self.params[0], int)
                or self.params[0] <= 0
            ):
                raise ValueError(
                    f"Parâmetro inválido para {self.type.name}: {self.params}. Esperado: [inteiro_positivo]."
                )
        elif self.type == IndicatorType.ADX:
            if (
                len(self.params) != 1
                or not isinstance(self.params[0], int)
                or self.params[0] <= 1
            ):
                raise ValueError(
                    f"Parâmetro inválido para {self.type.name}: {self.params}. Esperado: [inteiro > 1]."
                )
        elif self.type == IndicatorType.BB:
            if (
                len(self.params) != 2
                or not isinstance(self.params[0], int)
                or self.params[0] <= 0
                or not isinstance(self.params[1], (int, float))
                or self.params[1] <= 0
            ):
                raise ValueError(
                    f"Parâmetros inválidos para {self.type.name}: {self.params}. Esperado: [inteiro_positivo, numero_positivo]."
                )
        elif self.type == IndicatorType.MACD:
            if len(self.params) != 3 or not all(
                isinstance(p, int) and p > 0 for p in self.params
            ):
                raise ValueError(
                    f"Parâmetros inválidos para {self.type.name}: {self.params}. Esperado: [inteiro_positivo, inteiro_positivo, inteiro_positivo]."
                )
        elif self.type == IndicatorType.STOCH:
            if len(self.params) != 2 or not all(
                isinstance(p, int) and p > 0 for p in self.params
            ):
                raise ValueError(
                    f"Parâmetros inválidos para {self.type.name}: {self.params}. Esperado: [inteiro_positivo, inteiro_positivo]."
                )
        # Adicione validações para outros tipos conforme necessário

    def _generate_column_name(self) -> str:
        """Gera um nome de coluna/identificador único para esta configuração."""
        # Garante que params exista antes de usar
        if not hasattr(self, "params") or self.params is None:
            # Tenta buscar defaults se params não estiver definido (pode ocorrer se __init__ falhar antes)
            try:
                params_to_use = self._get_default_params(self.type)
            except ValueError:  # Se não há default e não foi dado, usa string vazia
                params_to_use = []
        else:
            params_to_use = self.params

        param_str = "_".join(
            map(
                lambda p: (
                    str(int(p)) if isinstance(p, float) and p.is_integer() else str(p)
                ),
                params_to_use,
            )
        )
        return f"{self.type.value}_{param_str}"

    @property
    def column_name(self) -> str:
        """Retorna o nome de coluna/identificador gerado."""
        # Garante que _column_name foi gerado
        if not hasattr(self, "_column_name"):
            self._column_name = self._generate_column_name()
        return self._column_name

    def __str__(self):
        return self.column_name

    def __repr__(self):
        return f"IndicatorConfig(type=IndicatorType.{self.type.name}, params={self.params})"

    def __eq__(self, other):
        if not isinstance(other, IndicatorConfig):
            return NotImplemented
        return self.type == other.type and self.params == other.params

    def __hash__(self):
        # Garante que params é uma tupla para hash
        params_tuple = tuple(self.params) if self.params is not None else tuple()
        return hash((self.type, params_tuple))
