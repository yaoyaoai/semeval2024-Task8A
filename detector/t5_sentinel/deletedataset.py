from langdetect import detect
import json

def detect_language(text):
    lang = detect(text)
    return lang
corpus=[]
data_dir= f"/home/yaoxy/T5-Sentinel-public-main/data/split/Semeval/SubtaskA/subtaskA_dev_multilingual.jsonl"
with open(data_dir, 'r') as f, open("no_ru_mt.jsonl", "w") as f1:
    for line in f:
        if detect_language( json.loads(line)['text'] ) != 'ru':
            # corpus.append(line)
            f1.write(line)
    
        