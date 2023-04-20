import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import string
import re

import scipy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc,f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchinfo

import time
import os
import shutil

import transformers
from transformers import AutoTokenizer

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from tqdm import tqdm
tqdm.pandas()

device="cuda"

mode = "ur"

train_stances = pd.read_csv("./dataset/train_stances.csv")
train_bodies = pd.read_csv("./dataset/train_bodies.csv")

test_stances = pd.read_csv("./dataset/competition_test_stances.csv")
test_bodies = pd.read_csv("./dataset/competition_test_bodies.csv")

test_df = test_stances.merge(test_bodies, on="Body ID")
test_df["Related"] = (test_df["Stance"] != "unrelated").astype(int)

train_and_val_df = train_stances.merge(train_bodies, on="Body ID")
train_and_val_df["Related"] = (train_and_val_df["Stance"] != "unrelated").astype(int)

if mode == "stance":
    train_and_val_df = train_and_val_df[train_and_val_df["Related"] == True]
    test_df = test_df[test_df["Related"] == True]

val_split_ratio = 0.2

def split_train_val(df, ratio):
    val_count = int(ratio * df["Body ID"].nunique())
    all_ids = list(df["Body ID"].unique())
    val_body_ids = random.sample(all_ids, val_count)
    train_body_ids = set(all_ids) - set(val_body_ids)
    
    assert len(set(val_body_ids) & train_body_ids) == 0
    
    val_df = df.loc[df["Body ID"].isin(val_body_ids)]
    train_df = df.loc[df["Body ID"].isin(train_body_ids)]
    
    return val_df, train_df

val_df, train_df = split_train_val(train_and_val_df, val_split_ratio)

def prepare_df(df):
    df = df.drop("Body ID", axis=1)
    df = df.reset_index()
    df = df.drop("index", axis=1)
    df["Related"] = df["Stance"] != "unrelated"
    return df

val_df = prepare_df(val_df)
train_df = prepare_df(train_df)
test_df = prepare_df(test_df)

# Most of this from the first practical
additional_specials = ["—", "”", "“", "’", "‘"]

def remove_excess_whitespace(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.strip()
    return text

def remove_punctuation(text):
    punc = str.maketrans('', '', string.punctuation)
    text = text.translate(punc)
    
    for special in additional_specials:
        text = text.replace(special, "")
    
    return text

def remove_urls(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub('', text)

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub('', text)

def remove_numbers(text):
    numbers = re.compile(r'\d+')
    return numbers.sub('', text)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)

def apply_cleaning(text, excess=True, punc=True, urls=True, html=True, numbers=True, emojis=True, lower=True):
    if excess:
        text = " ".join(text.split())
        
    if punc:
        text = remove_punctuation(text)
    
    if urls:
        text = remove_urls(text)
    
    if html:
        text = remove_html(text)
    
    if numbers:
        text = remove_numbers(text)
        
    if emojis:
        text = remove_emojis(text)
        
    if lower:
        text = text.lower()
    
    return text

config_remove_excess_whitespace = True
config_remove_punctuation = False
config_remove_urls = True
config_remove_html = True
config_remove_numbers = False
config_remove_emojis = True
config_convert_to_lowercase = False

def process_text(text):
    text = apply_cleaning(
        text, 
        excess=config_remove_excess_whitespace, 
        punc=config_remove_punctuation, 
        urls=config_remove_urls, 
        html=config_remove_html, 
        numbers=config_remove_numbers, 
        emojis=config_remove_emojis, 
        lower=config_convert_to_lowercase
    )
    
    return text

train_df["Processed Headline"] = train_df["Headline"].progress_apply(process_text)
train_df["Processed Body"] = train_df["articleBody"].progress_apply(process_text)

val_df["Processed Headline"] = val_df["Headline"].progress_apply(process_text)
val_df["Processed Body"] = val_df["articleBody"].progress_apply(process_text)

test_df["Processed Headline"] = test_df["Headline"].progress_apply(process_text)
test_df["Processed Body"] = test_df["articleBody"].progress_apply(process_text)

selected_model = "distilroberta-base" #"bert-base-uncased"
tokeniser = AutoTokenizer.from_pretrained(selected_model)

def concated_headline_body_tokens(headline, body):
    concated = tokeniser(headline, body, truncation="longest_first", padding="max_length", return_tensors="pt")
    input_ids = concated["input_ids"]
    attention_mask = concated["attention_mask"]
    return input_ids, attention_mask

# train_df = train_df[:10]
# val_df = val_df[:10]

transformers.logging.set_verbosity_error()
train_df[["input_ids", "attention_mask"]] = train_df.progress_apply(lambda row: concated_headline_body_tokens(row["Processed Headline"], row["Processed Body"]), axis="columns", result_type="expand") # type: ignore
transformers.logging.set_verbosity_warning()

transformers.logging.set_verbosity_error()
val_df[["input_ids", "attention_mask"]] = val_df.progress_apply(lambda row: concated_headline_body_tokens(row["Processed Headline"], row["Processed Body"]), axis="columns", result_type="expand") # type: ignore
transformers.logging.set_verbosity_warning()

mode = "ur"

if mode == "stance":
    labels2id = {
        "agree": 0,
        "disagree": 1,
        "discuss": 2
    }

    train_labels = np.array([labels2id[x] for x in train_df["Stance"].values])
    train_labels_tensor = torch.LongTensor(train_labels).unsqueeze(1)
    train_labels_tensor.shape

    val_labels = np.array([labels2id[x] for x in val_df["Stance"].values])
    val_labels_tensor = torch.LongTensor(val_labels).unsqueeze(1)
    val_labels_tensor.shape
elif mode == "ur":
    labels2id = {
        "Unrelated": 0,
        "Related": 1
    }

    train_labels = np.array([int(x) for x in train_df["Related"].values])
    train_labels_tensor = torch.LongTensor(train_labels).unsqueeze(1)
    train_labels_tensor.shape

    val_labels = np.array([int(x) for x in val_df["Related"].values])
    val_labels_tensor = torch.LongTensor(val_labels).unsqueeze(1)
    val_labels_tensor.shape
else:
    assert 1 == 0, "Error"

unique_class_labels = np.unique(train_labels)
class_weights = compute_class_weight("balanced", classes=unique_class_labels, y=train_labels)
class_weights_tensor = torch.from_numpy(class_weights)
print(class_weights_tensor)

# counts = [0, 0, 0]

# for label in train_labels:
#     counts[label] += 1

# print(counts)

# Create the confusion matrix - from Practical 1
def plot_confusion_matrix(y_test, y_pred):
    ''' Plot the confusion matrix for the target labels and predictions '''
    cm = confusion_matrix(y_test, y_pred)

    # Create a dataframe with the confusion matrix values
    df_cm = pd.DataFrame(cm, range(cm.shape[0]),
                  range(cm.shape[1]))

    # Plot the confusion matrix
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True,fmt='.0f',cmap="YlGnBu",annot_kws={"size": 10}) # font size
    plt.show()

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx] 
        }

        return item
    
    def __len__(self):
        return len(self.labels)

batch_size = 8

train_transformer_input_ids = torch.concat(list(train_df["input_ids"].values))
train_transformer_attention_masks = torch.concat(list(train_df["attention_mask"].values))
train_dataset = HFDataset(train_transformer_input_ids, train_transformer_attention_masks, train_labels_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_transformer_input_ids = torch.concat(list(val_df["input_ids"].values))
val_transformer_attention_masks = torch.concat(list(val_df["attention_mask"].values))
val_dataset = HFDataset(val_transformer_input_ids, val_transformer_attention_masks, val_labels_tensor)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import evaluate
from scipy.special import softmax

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=1)

    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=labels)
    precision = evaluate.load("precision").compute(predictions=predictions, references=labels, average="weighted")
    recall = evaluate.load("recall").compute(predictions=predictions, references=labels, average="weighted")
    f1 = evaluate.load("f1").compute(predictions=predictions, references=labels, average="weighted")

    if mode == "stance":
        roc_auc = evaluate.load("roc_auc", "multiclass").compute(prediction_scores=probs, references=labels.squeeze(), average="weighted", multi_class="ovr")
    else:
        roc_auc = 0 # evaluate.load("roc_auc").compute(prediction_scores=probs, references=labels.squeeze(), average="weighted")

    confusion_labels = [0, 1, 2] if mode == "stance" else [0, 1]

    confusion = metrics.confusion_matrix(labels.squeeze(), predictions, labels=confusion_labels)
    print(confusion)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}

# https://stackoverflow.com/q/70979844
class WeightedCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_func = nn.CrossEntropyLoss(weight=class_weights_tensor.float().to(device)).to(device)
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1)) # type: ignore

        return (loss, outputs) if return_outputs else loss

from sadice import SelfAdjDiceLoss 

# https://stackoverflow.com/q/70979844
class SelfAdjDiceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").view(-1)
        outputs = model(**inputs)
        logits = outputs.get("logits").view(-1, self.model.config.num_labels) # type: ignore
        loss_func = SelfAdjDiceLoss().to(device)

        # print(logits.shape, labels.shape)

        loss = loss_func(logits, labels)

        return (loss, outputs) if return_outputs else loss

model: torch.nn.Module = AutoModelForSequenceClassification.from_pretrained(
    selected_model,
    num_labels=3 if mode == "stance" else 2
) # type: ignore

training_args = TrainingArguments(f"test-trainer-{time.time()}", evaluation_strategy="epoch", num_train_epochs=10)

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokeniser,
    compute_metrics=compute_metrics
)

trainer.train()

# from torch.utils.data import DataLoader
# from transformers import BertForSequenceClassification

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model: torch.nn.Module = BertForSequenceClassification.from_pretrained(
#     selected_model,
#     num_labels=3
# ) # type: ignore
# model.to(device)
# model.train()

# opt = optim.AdamW(model.parameters(), lr=5e-5)

# for epoch in range(3):
#     for batch in tqdm(train_dataloader):
#         opt.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         opt.step()

# model.eval()