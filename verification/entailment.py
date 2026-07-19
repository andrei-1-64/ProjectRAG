import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CitationVerifier:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base"):
        print(f"Încărcăm modelul de verificare {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def verify(self, claim, docs):
        # Îmbinăm documentele într-un singur context (Premisa)
        context = "\n".join(docs)
        
        # FIX: Ordinea este CRUCIALĂ! Primul este Contextul (Premisa), al doilea este Claim-ul (Ipoteza)
        inputs = self.tokenizer(context, claim, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        probs = torch.softmax(logits, dim=1)
        
        # Indexul 1 este Entailment (Susținere/Adevăr) pentru DeBERTa v3 NLI
        entailment_score = probs[0][1].item()
        
        return entailment_score