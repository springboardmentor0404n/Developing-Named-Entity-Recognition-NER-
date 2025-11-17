import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score

NUMERIC_TOLERANCE = 0.05   # 5% tolerance


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_values(true_val, pred_val):
    """
    Returns 1 if match (within tolerance for numeric), else 0
    """
    # If numeric, use tolerance
    if isinstance(true_val, (int, float)) and isinstance(pred_val, (int, float)):
        if true_val == 0:
            return 1 if pred_val == 0 else 0
        error = abs(pred_val - true_val) / abs(true_val)
        return 1 if error <= NUMERIC_TOLERANCE else 0

    # String comparison
    return 1 if str(true_val).lower() == str(pred_val).lower() else 0


def evaluate(ground_truth_dir, prediction_dir):
    gt_files = sorted(os.listdir(ground_truth_dir))

    total_fields = 0
    correct_fields = 0

    # For precision/recall/F1
    y_true = []
    y_pred = []

    for file_name in gt_files:
        gt_path = os.path.join(ground_truth_dir, file_name)
        pr_path = os.path.join(prediction_dir, file_name)

        if not os.path.exists(pr_path):
            print(f"⚠️ Prediction missing for {file_name}")
            continue

        gt = load_json(gt_path)
        pred = load_json(pr_path)

        for key in gt.keys():
            total_fields += 1

            if key in pred:
                is_correct = compare_values(gt[key], pred[key])

                # Use binary to compute precision/recall/f1
                y_true.append(1)
                y_pred.append(is_correct)

                correct_fields += is_correct

            else:
                # Field missing → FN
                y_true.append(1)
                y_pred.append(0)

    accuracy = correct_fields / total_fields if total_fields > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    ACC, PREC, REC, F1 = evaluate("ground_truth", "predictions")

    print("\n📊 MODEL EVALUATION RESULTS")
    print("---------------------------")
    print(f"✅ Accuracy: {ACC:.4f}")
    print(f"✅ Precision: {PREC:.4f}")
    print(f"✅ Recall: {REC:.4f}")
    print(f"✅ F1 Score: {F1:.4f}")
