import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, jaccard_score
)
import numpy as np # Import numpy for placeholder data

# Placeholder data for demonstration. Replace with your actual y_test, y_pred, and y_prob.
y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
y_prob = np.array([0.1, 0.4, 0.2, 0.8, 0.3, 0.7, 0.6, 0.9, 0.25, 0.95])

# Metrics
f1 = f1_score(y_test, y_pred)
iou = jaccard_score(y_test, y_pred)

print("===== MODEL DASHBOARD ====")
print("F1 Score:", f1)
print("IoU Score:", iou)

# Confusion Matrix
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
plt.figure()
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

# Precision-Recall Curve
plt.figure()
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
