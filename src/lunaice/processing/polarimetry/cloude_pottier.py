from __future__ import annotations

import logging

import numpy as np

from lunaice.models.schemas import CoherencyMatrix, DecompositionProducts

logger = logging.getLogger(__name__)


class CloudePottierDecomposition:
    def decompose(self, t3: CoherencyMatrix) -> DecompositionProducts:
        h, w = t3.shape
        entropy = np.full((h, w), np.nan, dtype=np.float64)
        alpha_deg = np.full((h, w), np.nan, dtype=np.float64)
        anisotropy = np.full((h, w), np.nan, dtype=np.float64)
        lambda_1 = np.full((h, w), np.nan, dtype=np.float64)
        lambda_2 = np.full((h, w), np.nan, dtype=np.float64)
        lambda_3 = np.full((h, w), np.nan, dtype=np.float64)
        alpha_1_arr = np.full((h, w), np.nan, dtype=np.float64)
        alpha_2_arr = np.full((h, w), np.nan, dtype=np.float64)
        alpha_3_arr = np.full((h, w), np.nan, dtype=np.float64)

        for i in range(h):
            for j in range(w):
                T = np.array([
                    [t3.t11[i, j], t3.t12[i, j], t3.t13[i, j]],
                    [t3.t21[i, j], t3.t22[i, j], t3.t23[i, j]],
                    [t3.t31[i, j], t3.t32[i, j], t3.t33[i, j]],
                ], dtype=np.complex128)
                T = (T + T.conj().T) / 2.0
                eigenvalues, eigenvectors = np.linalg.eigh(T)
                eigenvalues = np.maximum(eigenvalues[::-1], 0)
                eigenvectors = eigenvectors[:, ::-1]
                l1, l2, l3 = eigenvalues
                lambda_1[i, j] = l1
                lambda_2[i, j] = l2
                lambda_3[i, j] = l3
                total = l1 + l2 + l3
                if total > 0:
                    p1, p2, p3 = l1 / total, l2 / total, l3 / total
                    p_safe = np.maximum([p1, p2, p3], 1e-30)
                    entropy[i, j] = -np.sum(p_safe * np.log(p_safe)) / np.log(3)
                else:
                    p1 = p2 = p3 = 1.0 / 3.0
                    entropy[i, j] = 0.0
                anisotropy[i, j] = (l2 - l3) / (l2 + l3 + 1e-30) if (l2 + l3) > 0 else 0.0
                for k_idx, (ev, eival) in enumerate(zip(eigenvectors.T, eigenvalues)):
                    eiv = ev / (np.linalg.norm(ev) + 1e-30)
                    k_pauli = np.array([1.0, 0.0, 0.0])
                    cos_alpha = np.abs(np.dot(eiv, k_pauli))
                    alpha_val = np.degrees(np.arccos(np.clip(cos_alpha, 0.0, 1.0)))
                    if k_idx == 0:
                        alpha_1_arr[i, j] = alpha_val
                    elif k_idx == 1:
                        alpha_2_arr[i, j] = alpha_val
                    else:
                        alpha_3_arr[i, j] = alpha_val
                alpha_deg[i, j] = p1 * alpha_1_arr[i, j] + p2 * alpha_2_arr[i, j] + p3 * alpha_3_arr[i, j]

        logger.info("Cloude-Pottier decomposition complete")
        return DecompositionProducts(
            entropy=entropy.astype(np.float32),
            alpha_deg=alpha_deg.astype(np.float32),
            anisotropy=anisotropy.astype(np.float32),
            lambda_1=lambda_1.astype(np.float32),
            lambda_2=lambda_2.astype(np.float32),
            lambda_3=lambda_3.astype(np.float32),
            alpha_1=alpha_1_arr.astype(np.float32),
            alpha_2=alpha_2_arr.astype(np.float32),
            alpha_3=alpha_3_arr.astype(np.float32),
        )
