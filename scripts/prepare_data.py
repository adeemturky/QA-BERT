import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizerFast

def read_squad(path):
    with open(path, 'r') as file:
        squad_dict = json.load(file)

    contexts, questions, answers = [], [], []

    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        else:
            for i in range(len(context)):
                if context[i:i+len(gold_text)] == gold_text:
                    answer['answer_start'] = i
                    answer['answer_end'] = i + len(gold_text)
                    break

def prepare_data(tokenizer, contexts, questions, answers):
    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []

    
    subset_size = min(1500, len(contexts))
    contexts = contexts[:subset_size]
    questions = questions[:subset_size]
    answers = answers[:subset_size]

    for context, question, answer in zip(contexts, questions, answers):
        inputs = tokenizer(
            question,
            context,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=384
        )

        offset_mapping = inputs.pop("offset_mapping")
        input_ids.append(inputs["input_ids"])
        attention_masks.append(inputs["attention_mask"])

        start_char = answer["answer_start"]
        end_char = answer["answer_end"]

        token_start_index = 0
        token_end_index = 0

        for i, (start, end) in enumerate(offset_mapping):
            if start <= start_char < end:
                token_start_index = i
            if start < end_char <= end:
                token_end_index = i
                break

        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    return {
        "input_ids": tf.convert_to_tensor(input_ids, dtype=tf.int32),
        "attention_mask": tf.convert_to_tensor(attention_masks, dtype=tf.int32),
        "start_positions": start_positions,
        "end_positions": end_positions
    }

if __name__ == '__main__':
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    contexts, questions, answers = read_squad('data/train-v1.1.json')
    add_end_idx(answers, contexts)

    train_contexts, val_contexts, train_questions, val_questions, train_answers, val_answers = train_test_split(
        contexts, questions, answers, test_size=0.1
    )

    train_data = prepare_data(tokenizer, train_contexts, train_questions, train_answers)
    val_data = prepare_data(tokenizer, val_contexts, val_questions, val_answers)

    print("Training data and validation data prepared.")