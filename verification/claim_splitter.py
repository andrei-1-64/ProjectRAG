import openai # Sau modelul local pe care îl folosești

class ClaimSplitter:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name

    def extract(self, answer: str):
        """
        Descompune răspunsul generat în afirmații atomice (fact-check-able claims).
        """
        prompt = f"""
        Ești un expert în analiză logică. Descompune următorul răspuns în afirmații factuale atomice. 
        Fiecare afirmație trebuie să fie completă (subiect + predicat) și să poată fi verificată independent.
        
        Răspuns: "{answer}"
        
        Output format: Listă JSON de string-uri.
        """
        
        # Aici facem apelul către LLM
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "Returnează doar un array JSON valid."},
                      {"role": "user", "content": prompt}]
        )
        
        # Presupunem că răspunsul este un string JSON pe care îl parsăm
        import json
        return json.loads(response.choices[0].message.content)