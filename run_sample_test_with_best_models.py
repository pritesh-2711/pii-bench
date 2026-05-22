from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch, json

model_path = "models/best_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

text = "My name is Pritesh Jha and my email is pritesh@example.com"

enc = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256,
    return_offsets_mapping=True,
)

offsets = enc.pop("offset_mapping")[0]
with torch.no_grad():
    logits = model(**enc).logits

pred_ids = logits.argmax(dim=-1)[0].tolist()
id2label = model.config.id2label

for pred_id, (start, end) in zip(pred_ids, offsets.tolist()):
    if start == end:
        continue
    label = id2label[pred_id]
    if label != "O":
        print(label, repr(text[start:end]))