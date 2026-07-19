import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")

class Generator:
    def __init__(self, config):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Încărcăm modelul local {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            clean_up_tokenization_spaces=False
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=50,
            max_length=None, # <--- ASTA elimină avertismentul rămas
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            return_full_text=False
        )

    def generate(self, query, docs):
        context_str = "\n".join(docs) if docs else "No sources provided."
        
        # --- PROMPT MODIFICAT PENTRU PROPOZIȚII COMPLETE ---
        messages = [
            {
                "role": "system", 
                "content": "You are a factual AI assistant. Answer the user's question using ONLY the provided Context. You MUST write your answer as a complete sentence. Do not invent details."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context_str}\n\nQuestion: {query}"
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            outputs = self.pipe(prompt, clean_up_tokenization_spaces=False)
            generated_text = outputs[0]["generated_text"].strip()
            return generated_text
        except Exception as e:
            return f"Eroare la generarea locală: {str(e)}"