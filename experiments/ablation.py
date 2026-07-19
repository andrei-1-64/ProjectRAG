import sys
import os
import yaml
import json


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.generator import Generator

# Adaugă folderul părinte la sys.path pentru a importa main


from main import run_pipeline

def run_ablation_study(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Inițializează generatorul pentru baseline
    generator_simple = Generator(config['generation'])
    
    with open('data/benchmark/test_set.json', 'r') as f:
        dataset = json.load(f)
    
    results = []
    
    for mode in ["full", "baseline"]:
        print(f"Rulăm experimentul în modul: {mode}")
        for entry in dataset:
            if mode == "baseline":
                # Baseline: fără RAG, doar modelul generativ
                ans = generator_simple.generate(entry['question'], docs=[""]) 
                conf = 1.0 
            else:
                # Full RoRAG-Trust: cu verificare și abstinență
                ans, conf = run_pipeline(entry['question'], config)
            
            results.append({
                "mode": mode,
                "question": entry['question'],
                "answer": ans,
                "is_faithful": conf if conf else 0.0
            })
            
    # Salvare rezultate
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Ablation Study completat. Rezultatele sunt în results/ablation_results.json")

if __name__ == "__main__":
    run_ablation_study("configs/config.yaml")