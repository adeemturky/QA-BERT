import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering, DistilBertTokenizerFast
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import prepare_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading model and tokenizer...")
model = TFDistilBertForQuestionAnswering.from_pretrained("models/distilbert-qa")
tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert-qa")

# Load validation data
contexts, questions, answers = prepare_data.read_squad('data/train-v1.1.json')
prepare_data.add_end_idx(answers, contexts)
_, val_contexts, _, val_questions, _, val_answers = prepare_data.train_test_split(
    contexts, questions, answers, test_size=0.1
)

val_data = prepare_data.prepare_data(tokenizer, val_contexts, val_questions, val_answers)

# Generate predictions
y_pred_start = []
y_pred_end = []

y_true_start = val_data['start_positions']
y_true_end = val_data['end_positions']

for i in range(len(val_questions)):
    input_ids = val_data['input_ids'][i:i+1]
    attention_mask = val_data['attention_mask'][i:i+1]

    if tf.reduce_sum(attention_mask) == 0:
        continue  # skip empty inputs

    outputs = model(input_ids, attention_mask=attention_mask)
    start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    end = tf.argmax(outputs.end_logits, axis=1).numpy()[0]

    y_pred_start.append(start)
    y_pred_end.append(end)

# Trim ground truth lists to match prediction length
y_true_start = y_true_start[:len(y_pred_start)]
y_true_end = y_true_end[:len(y_pred_end)]

# Compute F1 and EM
f1_start = f1_score(y_true_start, y_pred_start, average='micro')
f1_end = f1_score(y_true_end, y_pred_end, average='micro')
em_start = accuracy_score(y_true_start, y_pred_start)
em_end = accuracy_score(y_true_end, y_pred_end)

print(f"\n Evaluation Complete")
print(f"F1 Score (Start): {f1_start:.4f}, F1 Score (End): {f1_end:.4f}")
print(f"Exact Match (Start): {em_start:.4f}, Exact Match (End): {em_end:.4f}")

# Save and show detailed report
os.makedirs("reports", exist_ok=True)
report_start = classification_report(y_true_start, y_pred_start, output_dict=True)
report_end = classification_report(y_true_end, y_pred_end, output_dict=True)

pd.DataFrame(report_start).transpose().to_csv("reports/start_report.csv")
pd.DataFrame(report_end).transpose().to_csv("reports/end_report.csv")

# Confusion matrices
cm_start = confusion_matrix(y_true_start, y_pred_start)
cm_end = confusion_matrix(y_true_end, y_pred_end)

np.save("reports/cm_start.npy", cm_start)
np.save("reports/cm_end.npy", cm_end)

# Heatmaps
plt.figure(figsize=(10, 8))
sns.heatmap(cm_start, annot=False, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Start Positions")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("reports/cm_start_heatmap.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(cm_end, annot=False, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - End Positions")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("reports/cm_end_heatmap.png")
plt.close()

print("\n Reports saved to 'reports/' folder.")
