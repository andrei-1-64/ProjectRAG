import yaml
import logging
from retrieval.hybrid import HybridRetriever
from generation.generator import Generator
from verification.claim_splitter import ClaimSplitter
from verification.entailment import CitationVerifier
from verification.abstention import AbstentionPolicy
from evaluation.faithfulness import FaithfulnessEvaluator

# Configurare Logging - esențial pentru cercetare
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RoRAG-Trust")

def run_pipeline(query, config):
    # Inițializare module bazată pe config
    retriever = HybridRetriever(config['retrieval'])
    generator = Generator(config['generation'])
    splitter = ClaimSplitter()
    verifier = CitationVerifier(config['verification']['nli_model'])
    policy = AbstentionPolicy(config['verification']['abstention_threshold'])
    evaluator = FaithfulnessEvaluator(config['verification']['abstention_threshold'])

    logger.info(f"Procesare query: {query}")
    
    # 1. Recuperare
    docs = retriever.search(query)
    
    # 2. Generare
    answer = generator.generate(query, docs)
    
    # 3. Verificare (Novel contribution)
    claims = splitter.extract(answer)
    verifications = [verifier.verify(c, docs) for c in claims]
    
    # 4. Evaluare Faithfulness & Abstinență
    is_faithful, score = evaluator.validate_answer(verifications)
    
    if policy.should_abstain(verifications):
        logger.warning(f"Abstinență declanșată pentru: {query} (Scor: {score:.2f})")
        return "Îmi pare rău, dar informațiile sunt insuficiente sau contradictorii.", score
    
    logger.info(f"Răspuns validat (Scor: {score:.2f})")
    return answer, score

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Exemplu de apel
    ans, conf = run_pipeline("Cine a fondat Politehnica din București?", config)
    print(f"Rezultat: {ans} | Încredere: {conf}")