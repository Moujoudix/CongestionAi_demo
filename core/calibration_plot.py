from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
def save_calibration_plot(y_true, y_prob, out_path: str):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1],'--')
    plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency'); plt.title('Calibration')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
