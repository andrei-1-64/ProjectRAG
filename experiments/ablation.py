import yaml
import json
from main import run_pipeline

def run_ablation_study(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    with open('data/benchmark/test_set.json', 'r') as f:
        dataset = json.load(f)
    
    results = []
    
    # Rulăm experimentul cu și fără verificare (Abation)
    for mode in ["full", "baseline"]:
        print(f"Rulăm experimentul în modul: {mode}")
        for entry in dataset:
            if mode == "baseline":
                # Scenariu: ignorăm verificarea setând pragul la 0 sau bypass
                ans = generator_simple.generate(entry['question']) # Implementare fără verifier
                conf = 1.0 
            else:
                ans, conf = run_pipeline(entry['question'], config)
            
            results.append({
                "mode": mode,
                "question": entry['question'],
                "answer": ans,
                "is_faithful": conf if conf else 0.0
            })
            
    # Salvare rezultate pentru generarea graficelor în lucrare
    with open('results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Ablation Study completat. Rezultatele sunt în results/ablation_results.json")

if __name__ == "__main__":
    run_ablation_study("configs/config.yaml")