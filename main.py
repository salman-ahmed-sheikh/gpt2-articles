import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from google_trans_new import google_translator
import csv
import random


def translate(items):
    translator = google_translator()
    if type(items) == "list":
        ret = []        
        for item in items:
            ret.append(translator.translate(item, lang_tgt = 'hu'))
        return ret
    else:
        return translator.translate(items, lang_tgt = 'hu')
def selectRandom (items, minm, maxm):
    count = random.randint(minm, maxm)
    return random.sample(items, count)




tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")#gpt2-xl
model = TFGPT2LMHeadModel.from_pretrained("gpt2-xl", pad_token_id=tokenizer.eos_token_id)




output = csv.writer(open('output.csv', 'w',  encoding='utf-8'))

output.writerow(["keyword", "GUID", "Description", "Tags", "Article", "Category"])


# open title file
with open('titles.txt') as f:
    titles = f.readlines()

# open keywords file
with open('keywords.txt') as f:
    keywords = f.readlines()

# open title file
with open('images.txt') as f:
    images = f.readlines()

for i, (title, keyword) in enumerate (zip(titles, keywords)):
    #print(title, keyword)

    input_ids = tokenizer.encode(title, return_tensors='tf')
    beam_output = model.generate(
        input_ids, 
        max_length=1000, 
        num_beams=5, 
        early_stopping=True,
        temperature=0.7,
        top_k=50
    )

    print("Input ---- ", title)
    title = translate(title)
    keyword = translate(keyword)
    article = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    
    print(article)

    article = translate(article)
    article
    tags = translate(",".join(selectRandom(keywords,3,4)))
    categories = translate(",".join(selectRandom(keywords,1,2)))
    output.writerow([keyword, i+1, title, tags, article, categories])