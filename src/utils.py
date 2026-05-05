"""
공통 유틸리티 함수 모음
- PSD projection (eigenvalue floor)
- I/O 헬퍼
"""
import numpy as np
from pathlib import Path


def project_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """
    행렬을 Positive Semi-Definite로 projection
    1) symmetrize: (M + Mᵀ) / 2
    2) eigenvalue < floor → floor 으로 clip
    Reference: src/reference_code/20250812_refactored.py:75
    """
    M_sym = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals_clipped = np.clip(eigvals, floor, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
