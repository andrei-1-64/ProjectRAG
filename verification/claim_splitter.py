import os
import json
import re
from openai import OpenAI

class ClaimSplitter:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Inițializăm clientul doar dacă există o cheie validă
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def extract(self, answer: str):
        """
        Descompune răspunsul în afirmații atomice.
        Dacă nu există cheie OpenAI, folosește un splitter local bazat pe reguli.
        """
        if not self.client:
            # Abordare academică locală: spargem textul după punctuație (. ! ?)
            # Aceasta elimină costurile și erorile de credențiale
            claims = re.split(r'(?<=[.!?])\s+', answer.strip())
            return [c for c in claims if len(c.strip()) > 5]
        
        # Dacă există cheie, se execută apelul LLM normal
        prompt = f"""
        Ești un expert în analiză logică. Descompune următorul răspuns în afirmații factuale atomice. 
        Fiecare afirmație trebuie să fie completă (subiect + predicat) și să poată fi verificată independent.
        Răspuns: "{answer}"
        Output format: Listă JSON de string-uri.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Returnează doar un array JSON valid."},
                    {"role": "user", "content": prompt}
                ]
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            # Fallback în caz de eroare de rețea sau API limit
            claims = re.split(r'(?<=[.!?])\s+', answer.strip())
            return [c for c in claims if len(c.strip()) > 5]