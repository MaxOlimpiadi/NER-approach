# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 23:55:10 2026

@author: zidan
"""


import os
import json 
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from evaluate import load
from seqeval.metrics import accuracy_score

def token_labeling(input_folder):
    #categories = ['agentive', 'low_agentive', 'passive']
    categories = ['low_agentive']
    
    dict_labels = {'agentive': 'AG', 'low_agentive': 'LA', 'passive': 'PS'}
    
    all_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    print(all_files)
    
    
    with open('output.jsonl', 'w', encoding='utf-8') as fout: # open ouput file
        for file_name in all_files:
            file_path = os.path.join(input_folder, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as fin: # open current input file (chunk)
                data = json.load(fin)
                            
                tokens = [t['text'] for t in data["tokens"]]
                labels = ['O'] * len(tokens)
                
                #getting proper labels for tokens of the chunk:
                for ann in data["annotations"]:
                    for category in categories:
                        
                        # check whether the whole category toekns are included into the phrase:
                        if not set(ann[category]['tokenIds']).issubset(ann['phrase']['tokenIds']):
                            continue # if not - omit it
                            
                        if ann[category]['tokenIds']:
                            categoryTokenIds = sorted(ann[category]['tokenIds']) # token Ids of the category in the annotation
                            
                            
                            
                            # B-:
                            B_token = categoryTokenIds[0]   # token for B-label
                            prev = B_token 
                            if labels[B_token] == 'O': # if the label still hasn`t been changed yet (по дефолту же все лейблы 'O' в начале)
                                labels[B_token] = f'B-{dict_labels[category]}'   # gonna have B-AG or B-LA or B-PS
                            else:
                                continue # не размечаем этот спан вообще, иначе будут I- без B-
                            
                            # I-:
                            for tid in categoryTokenIds[1:]:     #from the second elemnt, cause first one has been already processed on the previous line
                                if labels[tid] == 'O':  # only if the corresponding label was not edited yet
                                    if tid == prev + 1: # if there is no hole betweet tokens like [73, 75] 
                                        labels[tid] = f'I-{dict_labels[category]}' # I-AG, I-LA or I-PS
                                    else: # then there is a hole, so we have to put B-..
                                        labels[tid] = f'B-{dict_labels[category]}' 
                                    prev = tid
                                
                # creating a corresponding json-line:
                record = {
                    'documentName': data['documentName'],
                    'tokens': tokens,
                    'labels': labels
                }                   
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                

def make_tokenize_and_align_labels(tokenizer, label2id):
    
    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True
        )
    
        new_labels = []
        for i, labels in enumerate(examples["labels"]):
            word_ids = tokenized.word_ids(batch_index=i) # получаем маску соответствия сабтокенов к словам: [None, 0, 0, None] - Первое слово 2 сабтокена имеет
            previous_word_idx = None # это надо чтобы момент начала нового слова контролить
            label_ids = [] # наши итоговые лейблы для сабтокенов уже (для текущего примера в examples).
    
            for word_idx in word_ids: # перебираем нашу маску сабтокенов. Дальше изи.
                if word_idx is None:
                    label_ids.append(-100)   # special tokens ignored
                elif word_idx != previous_word_idx: # типа если новое слово началось
                    label_ids.append(label2id[labels[word_idx]]) # word_idx - всегда указывает на исходный токен. Стало быть и лейбл его мы изи вытягивем. Но лейбл именно в цифровом формате для HF!
                else:
                    # # тот же токен → продолжение
                    # if labels[word_idx].startswith("B-"):
                    #     label_ids.append(label2id["I-" + labels[word_idx][2:]])
                    # else:
                    #     label_ids.append(label2id[labels[word_idx]])
                    label_ids.append(-100)
    
                previous_word_idx = word_idx
    
            new_labels.append(label_ids) # единый список спиской лейблов для батча
    
        tokenized["labels"] = new_labels
        return tokenized
    
    return tokenize_and_align_labels
    
            
            

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)

    true_preds = []
    true_labels = []
    
    
    label_list = [
    "O",
    "B-LA", "I-LA"
    ]
    
    label2id = {l: i for i, l in enumerate(label_list)} # 'O' -> 0, 'B-AG' -> 1, ...
    id2label = {i: l for l, i in label2id.items()} # 0 -> 'O', 1 -> 'B-AG', ...

    for pred_seq, label_seq in zip(preds, labels):
        seq_preds = []
        seq_labels = []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id == -100:
                continue
            seq_preds.append(id2label[p_id])
            seq_labels.append(id2label[l_id])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds)
    }

    



        
            
            
def main():
    # token_labeling('raw_data')
    
    
    INPUT_FOLDER = 'one-label-processed-data/low_agentive'

    # заряжаем токенайзер:
    model_name = 'LSX-UniWue/ModernGBERT_134M'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # снаряжаем датаколлатор, чтобы автоматически выравнивал тензоры не нулями, а -100
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)    

    # подготавливаем инфу о лейблах в нужных форматах:
    label_list = [
    "O",
    "B-LA", "I-LA"
    ]
    
    label2id = {l: i for i, l in enumerate(label_list)} # 'O' -> 0, 'B-AG' -> 1, ...
    id2label = {i: l for l, i in label2id.items()} # 0 -> 'O', 1 -> 'B-AG', ...
    
    
    # собираем пути к частям нашего датасета:
    train_path = os.path.join(INPUT_FOLDER, 'train.jsonl')
    dev_path = os.path.join(INPUT_FOLDER, 'dev.jsonl')
    test_path = os.path.join(INPUT_FOLDER, 'test.jsonl')
    
    #upload the dataset (in appropriate for HF format):
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "validation": dev_path,
        "test": test_path
    })
    
    
    
#--------------------Tokenization and alignment---------------------------------   
    # call the function "tokenize_and_align_labels" for every instance in dataset
    # в итоге получаем обновлённый датасет уже. А batched = true - для оптизимации
    fabricated_function = make_tokenize_and_align_labels(tokenizer, label2id)
    tokenized_dataset = dataset.map(fabricated_function, batched=True)

    #print(tokenized_dataset["train"][0])
#------------------------------------------------------------------------------


    # Creating the model:
    model = AutoModelForTokenClassification.from_pretrained( # просто добавляем к претрейнед выбранной модели ещё 1 голову (слой), которую мы будем обучать через NER)
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    
    
    #Specify the config for the training:
    training_args = TrainingArguments (
        output_dir = "./outputs_single_label",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        #save_strategy = "no",
        learning_rate = 2e-5,
        weight_decay = 0.01,
        num_train_epochs = 3,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        #logging_steps = 50,
        logging_steps = 10, # было 10
        load_best_model_at_end = True,
        #load_best_model_at_end = False,
        metric_for_best_model = "f1",
        greater_is_better = True,
        save_total_limit = 2,
        logging_strategy="steps",
        # for experiment:
        disable_tqdm=True,
        report_to="none",
    )
    
    trainer = Trainer (
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        #tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # можешь убрать, если пока не хочешь метрики
    )
    
    
    # Implementing training process:
    train_output = trainer.train()
    print("TRAIN OUTPUT:", train_output)
    
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print("TEST METRICS:", test_metrics)
        
    
    
    
    
    

    

    
    
    
    
main()

