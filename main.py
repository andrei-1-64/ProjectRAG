# main.py completat

import yaml
from utils.logger import setup_logger
from retrieval.hybrid import HybridRetriever
from generation.generator import Generator
from verification.claim_splitter import ClaimSplitter
from verification.entailment import CitationVerifier
from verification.abstention import AbstentionPolicy

# 1. Încarcă configurația
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Inițializează modulele
retriever = HybridRetriever(config['retrieval'])
generator = Generator(config['generation'])
verifier = CitationVerifier(config['verification'])
abstention_policy = AbstentionPolicy(config['verification']['abstention_threshold'])
logger = setup_logger("RoRAG")

def run_pipeline(query):
    logger.info(f"Query: {query}")
    
    # 1. Recuperare
    docs = retriever.search(query)
    
    # 2. Generare
    answer = generator.generate(query, docs)
    
    # 3. Verificare (Novel contribution)
    claims = claim_splitter.extract(answer)
    # Stocăm rezultatele detaliate pentru raport (de ex: scorul pentru fiecare claim)
    verification_details = [{"claim": c, "score": verifier.verify(c, docs)} for c in claims]
    results = [d["score"] for d in verification_details]
    
    # 4. Decizie și LOGGING
    if abstention_policy.should_abstain(results):
        logger.warning(f"Abstention triggered for query: {query}. Details: {verification_details}")
        return "Îmi pare rău, dar nu pot verifica acuratețea acestui răspuns.", None
    
    logger.info(f"Answer returned: {answer}")
    return answer, verification_details