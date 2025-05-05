import streamlit as st
import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering, DistilBertTokenizerFast

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = TFDistilBertForQuestionAnswering.from_pretrained("adeem6/distilbert-qa")
    tokenizer = DistilBertTokenizerFast.from_pretrained("adeem6/distilbert-qa")
    return model, tokenizer

model, tokenizer = load_model()

st.title("🧠 BERT Question Answering")
st.markdown("Enter a context and a question, and we'll extract the answer for you!")

context = st.text_area("📄 Context", height=200)
question = st.text_input("❓The question")

if st.button("🔍 Get an answer"):
    if not context or not question:
        st.warning("Please include both context and question.")
    else:
        inputs = tokenizer(
            question,
            context,
            return_tensors="tf",
            truncation=True,
            padding="max_length",
            max_length=384
        )

        outputs = model(inputs)
        start_idx = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
        end_idx = tf.argmax(outputs.end_logits, axis=1).numpy()[0]

        input_ids = inputs["input_ids"][0].numpy()
        answer = tokenizer.decode(
        input_ids[start_idx:end_idx+1],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
)


        st.success(f"🗣️ The answer: {answer}")
