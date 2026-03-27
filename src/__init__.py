"""
ABPM Hemodynamic Uncoupling Analysis Package
=============================================

A modular package for analyzing hemodynamic coupling patterns in ambulatory
blood pressure monitoring (ABPM) data.

Authors: Vitalii Babenko and Alyona Tymchak
Refactored Date: 2025-12-22
"""

__version__ = "2.0.0"
__author__ = "Vitalii Babenko and Alyona Tymchak"

from .config import Columns, Config
from .models import ModelPerformance, StatisticalResult, SubjectResult

__all__ = [
    "Config",
    "Columns",
    "SubjectResult",
    "ModelPerformance",
    "StatisticalResult",
]
