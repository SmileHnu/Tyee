import torch
import torch.nn as nn
import torch.optim as optim

from dataset import RandomDataset,CarDataset
from models.downstream.emotion_recognition_model import MLP
from trainer import Trainer
from tasks.emotion_recognition_task import EmotionRecognitionTask



def main():
    # dataset = RandomDataset(num_samples=1000, input_dim=10, num_classes=2)
    dataset = CarDataset()
    
    # model = MLP(input_dim=10, hidden_dim=64, output_dim=2)
    model = MLP(input_dim=6, hidden_dim=64, output_dim=4)
    optimizer = optim.Adam(model.parameters(),lr = 0.005)
    loss = nn.CrossEntropyLoss()
    task = EmotionRecognitionTask(dataset=dataset, model=model, optimizer=optimizer, loss=loss)
    trainer = Trainer(task=task, prop=0.8, batch_size=40)
    trainer.train(total_epochs=100, valid_epoch=10, save_epoch=10)




if __name__ == "__main__":
    main()