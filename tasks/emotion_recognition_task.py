from . import PRLTask

class EmotionRecognitionTask(PRLTask):
    def __init__(self, cfg, rank, world_size):
        super().__init__(cfg, rank, world_size)
        
    def train_step(self, data, target):
        # print(type(data),type(target))
        data, target= data.to(self.rank), target.to(self.rank)
        output,_ = self.model(data, data.size(1))
        loss = self.loss(output,target)

        return {
            'loss':loss
        }
        
    def valid_step(self, data, target):
        # print(type(data),type(target))
        output, _ = self.model(data)
        loss = self.loss(output,target)

        return {
            'loss':loss
        }
    
    def train(self,) -> None:
        self.model.train()
    
    def eval(self,) -> None:
        self.model.eval()
    
    
    

        