"""Visualization helpers for Gemini logical qubits and compiled move programs.

``ArchVisualizer`` (``bloqade.gemini.visualize.arch``) draws an architecture on
its own, statically with matplotlib or interactively with plotly; the debuggers
draw on top of it.
"""

from .debug import debugger as debugger, animated_debugger as animated_debugger
from .steane import render_steane_code_qubit as render_steane_code_qubit
from .plotly_debug import plotly_debugger as plotly_debugger
