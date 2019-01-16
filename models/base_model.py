class BaseModel:
    def __init__(self,):
        self.is_training = True

    def create_model(self, **unused_params):
        raise NotImplementedError()

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False
