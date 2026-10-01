from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from .base_tokenizer import BaseTokenizer


class WordLevelTokenizer(BaseTokenizer):

    def __init__(self, vocab_size=20_000, min_frequency=2):
        super().__init__()
        self.tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()

        self.trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<PAD>", "<UNK>"]
        )

    def train(self, texts):
        self.tokenizer.train_from_iterator(
            texts,
            trainer= self.trainer,
        )

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id("<PAD>")
    
    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)