import streamlit as st
from transformers import PhobertTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')
model= AutoModelForSequenceClassification.from_pretrained('vinai/phobert-base', num_labels=7,ignore_mismatched_sizes=True)
model.to('cpu')
model.load_state_dict(torch.load(f'phobert_trained_best.pth',map_location=device)) #File weight



def infer(text, tokenizer,model , max_len=120):
  class_names = ['Enjoyment', 'Disgust', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Other']
  token = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', truncation=True,max_length=max_len, return_tensors='pt')

  input_ids = token['input_ids'].to(device)
  attention_mask = token['attention_mask'].to(device)

  outputs = model(input_ids=input_ids, attention_mask=attention_mask)
  _, y_pred = torch.max(outputs.logits, 1)
  return class_names[y_pred]


st.title("Sentiment Analysis")


input_str = st.text_input("Sentence:")


if st.button("Analyze"):
    output_str = input_str[::-1]  
    st.write("Sentiment: ", infer(input_str,tokenizer,model ))