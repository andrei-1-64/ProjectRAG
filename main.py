import yaml
import logging
from retrieval.hybrid import HybridRetriever
from generation.generator import Generator
from verification.claim_splitter import ClaimSplitter
from verification.entailment import CitationVerifier
from verification.abstention import AbstentionPolicy
from evaluation.faithfulness import FaithfulnessEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RoRAG-Trust")

def run_pipeline(query, config):
    retriever = HybridRetriever(config['retrieval'])
    generator = Generator(config['generation'])
    splitter = ClaimSplitter()
    verifier = CitationVerifier(config['verification']['nli_model'])
    policy = AbstentionPolicy(config['verification']['abstention_threshold'])
    evaluator = FaithfulnessEvaluator(config['verification']['abstention_threshold'])

    logger.info(f"Procesare query: {query}")
    
    docs = retriever.search(query)
    print(f"\n[DEBUG] Documente recuperate din baza de date: {docs}")
    
    answer = generator.generate(query, docs)
    print(f"\n[DEBUG] Răspunsul brut al lui Qwen: {answer}")
    
    claims = splitter.extract(answer)
    
    # --- FIX-UL PENTRU RĂSPUNSURI SCURTE ---
    if not claims or len(claims) == 0:
        claims = [answer]
    # ---------------------------------------
    
    print(f"[DEBUG] Afirmații extrase pentru verificare: {claims}")
    
    verifications = []
    for c in claims:
        score = verifier.verify(c, docs)
        print(f"[DEBUG] Scor Entailment pentru '{c}': {score:.4f}")
        verifications.append(score)
    
    is_faithful, score = evaluator.validate_answer(verifications)
    
    if policy.should_abstain(verifications):
        logger.warning(f"Abstinență declanșată pentru: {query} (Scor: {score:.2f})")
        return "Îmi pare rău, dar informațiile sunt insuficiente sau contradictorii.", score
    
    logger.info(f"Răspuns validat (Scor: {score:.2f})")
    return answer, score

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    ans, conf = run_pipeline("In ce an a fost infiintata politehnica bucuresti?", config)
    print(f"\nRezultat final: {ans} | Încredere: {conf:.4f}")