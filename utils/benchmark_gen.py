import json
import os

def generate_initial_benchmark():
    benchmark_data = [
        {
            "id": 1,
            "question": "Cine a fondat Politehnica din București?",
            "context": "Universitatea Politehnica din București a fost înființată în 1818 de Gheorghe Lazăr.",
            "expected_type": "supported",
            "ground_truth": "Gheorghe Lazăr"
        },
        {
            "id": 2,
            "question": "Care este capitala Japoniei?",
            "context": "Context irelevant despre istoria tehnologiei în România.",
            "expected_type": "unanswerable",
            "ground_truth": None
        },
        {
            "id": 3,
            "question": "Este Bucureștiul în Franța?",
            "context": "Document A: București este în România. Document B: București este în Franța.",
            "expected_type": "contradictory",
            "ground_truth": None
        }
    ]
    
    os.makedirs('data/benchmark', exist_ok=True)
    with open('data/benchmark/test_set.json', 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=4, ensure_ascii=False)
    
    print("Benchmark generat cu succes în data/benchmark/test_set.json")

if __name__ == "__main__":
    generate_initial_benchmark()