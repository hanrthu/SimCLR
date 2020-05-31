import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from models.resnet_simclr import ResNetSimCLR
import numpy as np
from sklearn import preprocessing
import os

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using device:", device)
# checkpoints_folder = os.path.join(folder_name, 'checkpoints')
# config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))

def get_cifar10_data_loaders(download, shuffle=False, batch_size=1024):
  train_dataset = datasets.CIFAR10('../data', download=True,train=True,
                                       transform=transforms.ToTensor())
  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=8, drop_last=False, shuffle=shuffle)

  test_dataset = datasets.CIFAR10('../data', download=True,train=False,
                                       transform=transforms.ToTensor())
        
  test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=8, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader


class ResNetFeatureExtractor(object):
  def __init__(self,config):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.config = config
    self.model = _load_resnet_model(self.device,self.config)

  def _inference(self, loader):
    feature_vector = []
    labels_vector = []
    for batch_x, batch_y in loader:

      batch_x = batch_x.to(self.device)
      labels_vector.extend(batch_y)

      features, _ = self.model(batch_x)
      feature_vector.extend(features.cpu().detach().numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

  def get_resnet_features(self):
    train_loader, test_loader = get_cifar10_data_loaders(download=True)
    X_train_feature, y_train = self._inference(train_loader)
    X_test_feature, y_test = self._inference(test_loader)

    return X_train_feature, y_train, X_test_feature, y_test


def _load_resnet_model(device,config):
  # Load the neural net module
  model = ResNetSimCLR(**config['model'])
  model.eval()
  checkpoints_folder = os.path.join('../runs', config['fine_tune_from'], 'checkpoints')
  state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
  model.load_state_dict(state_dict)
  model = model.to(device)
  return model

class LogiticRegressionEvaluator(object):
  def __init__(self, n_features, n_classes):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.log_regression = LogisticRegression(n_features, n_classes).to(self.device)
    self.scaler = preprocessing.StandardScaler()

  def _normalize_dataset(self, X_train, X_test):
    print("Standard Scaling Normalizer")
    self.scaler.fit(X_train)
    X_train = self.scaler.transform(X_train)
    X_test = self.scaler.transform(X_test)
    return X_train, X_test

  @staticmethod
  def _sample_weight_decay():
    # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
    weight_decay = np.logspace(-6, 5, num=45, base=10.0)
    weight_decay = np.random.choice(weight_decay)
    print("Sampled weight decay:", weight_decay)
    return weight_decay

  def eval(self, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
      self.log_regression.eval()
      for batch_x, batch_y in test_loader:
          batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
          logits = self.log_regression(batch_x)

          predicted = torch.argmax(logits, dim=1)
          total += batch_y.size(0)
          correct += (predicted == batch_y).sum().item()

      final_acc = 100 * correct / total
      print("Final Accuracy: %f" %(final_acc))
      self.log_regression.train()
      return final_acc


  def create_data_loaders_from_arrays(self, X_train, y_train, X_test, y_test):
    X_train, X_test = self._normalize_dataset(X_train, X_test)

    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).type(torch.long))
    train_loader = torch.utils.data.DataLoader(train, batch_size=1024, shuffle=False)

    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).type(torch.long))
    test_loader = torch.utils.data.DataLoader(test, batch_size=1024, shuffle=False)
    return train_loader, test_loader

  def train(self, X_train, y_train, X_test, y_test):
    
    train_loader, test_loader = self.create_data_loaders_from_arrays(X_train, y_train, X_test, y_test)

    weight_decay = self._sample_weight_decay()

    optimizer = torch.optim.Adam(self.log_regression.parameters(), 1e-3, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0

    for e in range(200):
      
      for batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

        optimizer.zero_grad()

        logits = self.log_regression(batch_x)

        loss = criterion(logits, batch_y)
        # print("Loss %f" %(loss))
        loss.backward()
        optimizer.step()

      epoch_acc = self.eval(test_loader)
      print("Accuracy: %f" %(epoch_acc))
      
      if epoch_acc > best_accuracy:
        print("Saving new model with accuracy {}".format(epoch_acc))
        best_accuracy = epoch_acc
        torch.save(self.log_regression.state_dict(), 'log_regression.pth')

    print("--------------")
    print("Done training")
    print("Best accuracy:", best_accuracy)



class LogisticRegression(nn.Module):
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)