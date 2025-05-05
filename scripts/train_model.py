import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering, DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import prepare_data  # assumes this script is in the same folder or PYTHONPATH

print(" Loading tokenizer and model...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

print("Preparing training and validation data...")
contexts, questions, answers = prepare_data.read_squad('data/train-v1.1.json')
prepare_data.add_end_idx(answers, contexts)

train_contexts, val_contexts, train_questions, val_questions, train_answers, val_answers = train_test_split(
    contexts, questions, answers, test_size=0.1
)

train_data = prepare_data.prepare_data(tokenizer, train_contexts, train_questions, train_answers)
val_data = prepare_data.prepare_data(tokenizer, val_contexts, val_questions, val_answers)

# Convert start/end positions to tensors
train_data["start_positions"] = tf.convert_to_tensor(train_data["start_positions"], dtype=tf.int32)
train_data["end_positions"] = tf.convert_to_tensor(train_data["end_positions"], dtype=tf.int32)
val_data["start_positions"] = tf.convert_to_tensor(val_data["start_positions"], dtype=tf.int32)
val_data["end_positions"] = tf.convert_to_tensor(val_data["end_positions"], dtype=tf.int32)

print("Starting training...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

history = model.fit(
    x={"input_ids": train_data["input_ids"], "attention_mask": train_data["attention_mask"]},
    y={"start_positions": train_data["start_positions"], "end_positions": train_data["end_positions"]},
    validation_data=(
        {"input_ids": val_data["input_ids"], "attention_mask": val_data["attention_mask"]},
        {"start_positions": val_data["start_positions"], "end_positions": val_data["end_positions"]},
    ),
    epochs=4,
    batch_size=4
)

model.save_pretrained("models/distilbert-qa")
tokenizer.save_pretrained("models/distilbert-qa")
print(" Model training complete and saved.")
