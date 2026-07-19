import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Generator:
    def __init__(self, config):
        # Alegem un model extrem de ușor, capabil să ruleze pe CPU
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Încărcăm modelul local {self.model_name}... (ar putea dura puțin la prima rulare pentru a-l descărca)")
        
        # Încărcăm Tokenizer-ul și Modelul
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Încărcăm modelul forțat pe CPU pentru a evita erori dacă nu ai GPU configurat
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32, 
            device_map="cpu" 
        )
        
        # Creăm pipeline-ul de generare
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True
        )

    def generate(self, query, docs):
        context_str = "\n\n".join(docs) if docs else "Fără surse suplimentare."
        
        # Construim mesajele în formatul standard de chat
        messages = [
            {"role": "system", "content": f"Ești un asistent strict. Răspunde scurt, doar pe baza surselor oferite.\nSurse: {context_str}"},
            {"role": "user", "content": query}
        ]
        
        # Aplicăm template-ul specific Qwen
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            outputs = self.pipe(prompt)
            # Textul generat include și prompt-ul, așa că îl decupăm pentru a returna doar răspunsul curat
            generated_text = outputs[0]["generated_text"][len(prompt):].strip()
            return generated_text
        except Exception as e:
            return f"Eroare la generarea locală: {str(e)}"