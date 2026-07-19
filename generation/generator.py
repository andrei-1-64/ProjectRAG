import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Generator:
    def __init__(self, config):
        
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Încărcăm modelul local {self.model_name}... (ar putea dura puțin la prima rulare pentru a-l descărca)")
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32, 
            device_map="cpu" 
        )
        
        
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
        
        
        messages = [
            {"role": "system", "content": f"Ești un asistent strict. Răspunde scurt, doar pe baza surselor oferite.\nSurse: {context_str}"},
            {"role": "user", "content": query}
        ]
        
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            outputs = self.pipe(prompt)
            
            generated_text = outputs[0]["generated_text"][len(prompt):].strip()
            return generated_text
        except Exception as e:
            return f"Eroare la generarea locală: {str(e)}"