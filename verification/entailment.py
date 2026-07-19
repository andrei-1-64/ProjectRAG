from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CitationVerifier:
    def __init__(self, model_path="roberta-base-nli"):
       
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def verify(self, claim, docs):
      
        evidence = " ".join(docs)
        inputs = self.tokenizer(claim, evidence, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        
        probs = torch.softmax(logits, dim=1)
        return probs[0][1].item()