class Generator:
    def __init__(self, config):
        self.model_name = config.get('model_name', 'gpt-4o')
        self.temperature = config.get('temperature', 0.0)
        # Aici poți inițializa clientul OpenAI sau modelul local (ex: HuggingFace pipeline)

    def generate(self, query, docs):
        """
        Generează un răspuns bazat pe interogare și documentele recuperate.
        """
        # Crearea contextului din documentele recuperate
        context_str = "\n\n".join(docs)
        
        # Aici va veni logica de apel către LLM
        prompt = f"Folosește următoarele informații pentru a răspunde la întrebare:\n\n{context_str}\n\nÎntrebare: {query}"
        
        # Exemplu simulat de răspuns (înlocuiește cu apelul real către API-ul tău)
        return "Acesta este un răspuns generat pe baza surselor."

    def __call__(self, query, docs):
        return self.generate(query, docs)