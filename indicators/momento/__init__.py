"""
Subpacote de Indicadores de Momento.

Este subpacote contém implementações de indicadores de momento/osciladores,
incluindo RSI, ROC, Estocástico e CCI.
"""

from .cci import CCIIndicator
from .roc import ROCIndicator
# Importar indicadores específicos
from .rsi import RSIIndicator
from .stochastic_oscillator import StochasticOscillatorIndicator

# Controla o que é exposto com __all__
__all__ = [
    "RSIIndicator",
    "ROCIndicator",
    "StochasticOscillatorIndicator",
    "CCIIndicator",
]
