class FaithfulnessEvaluator:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def compute_faithfulness_score(self, verification_results):
        """
        verification_results: listă de scoruri (float) primite de la CitationVerifier
        """
        if not verification_results:
            return 0.0
        
        # Media scorurilor de suport (Entailment) pentru toate claim-urile
        avg_score = sum(verification_results) / len(verification_results)
        return avg_score

    def validate_answer(self, verification_results):
        """
        Returnează (is_faithful, score)
        """
        score = self.compute_faithfulness_score(verification_results)
        is_faithful = score >= self.threshold
        return is_faithful, score