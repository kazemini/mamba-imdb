from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
     
     @abstractmethod
     def train(self, texts):
          pass
     
     @abstractmethod
     def tokenize(self, text):
          pass
     
     @abstractmethod
     def encode(self, text):
          pass
     
     @abstractmethod
     def decode(self, ids):
          pass
     
     @property
     @abstractmethod
     def vocab_size(self):
          pass