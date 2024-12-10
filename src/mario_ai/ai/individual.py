class Individual:
    def __init__(self, neural_network, score=0.0):
        self.neural_network = neural_network
        self.score = score

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def get_neural_network(self):
        return self.neural_network
