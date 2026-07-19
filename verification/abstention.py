class AbstentionPolicy:
    def __init__(self, threshold):
        self.threshold = threshold
    def should_abstain(self, scores):
        return sum(scores) / len(scores) < self.threshold if scores else True