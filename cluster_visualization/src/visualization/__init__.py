"""
Visualization modules for cluster detection data.

This package contains modules for creating Plotly traces, figures,
and handling all visualization-related operations.
"""

from .figures import FigureManager
from .traces import TraceCreator
from .trace_registry import TraceRegistry, TraceType

__all__ = ["TraceCreator", "FigureManager", "TraceRegistry", "TraceType"]
