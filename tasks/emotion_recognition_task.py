from . import PRLTask
from utils import lazy_import_module
from nn import UpstreamDownstreamModel

class EmotionRecognitionTask(PRLTask):
    def __init__(self, cfg):
        
        super().__init__(cfg)
        
    def build_model(self):
        up_module = lazy_import_module('models.upstream', self.upstream_select)
        uptream = up_module()

        down_module = lazy_import_module('models.downstream',self.downstream_select)
        downstream = down_module(classes=self.downstream_classes)

        return UpstreamDownstreamModel(uptream, downstream, self.upstream_trainable)

    def train_step(self, model, data, target):
        # print(type(data),type(target))
        
        output,_ = model(data, data.size(1))
        # print(target.device)
        # print(self.loss.weight.device)
        self.loss.weight = self.loss.weight.to(target.device)
        loss = self.loss(output,target)

        return {
            'loss':loss
        }
        
    def valid_step(self, model, data, target):
        # print(type(data),type(target))
        output, _ = model(data, data.size(1))
        self.loss.weight = self.loss.weight.to(target.device)
        loss = self.loss(output,target)

        return {
            'loss':loss
        }
    
    def train(self,) -> None:
        self.model.train()
    
    def eval(self,) -> None:
        self.model.eval()
    
    
    

        