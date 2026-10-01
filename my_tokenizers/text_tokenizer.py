from .word_level_tokenizer import WordLevelTokenizer


class TextTokenizer:
    def __init__(self, algorithm="wordlevel", vocab_size=20_000, min_frequency=2):
        self.select_algorithm(algorithm, vocab_size, min_frequency)

    def select_algorithm(
        self, algorithm="wordlevel", vocab_size=20_000, min_frequency=2
    ):
        if algorithm == "wordlevel":
            self.tokenizer = WordLevelTokenizer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
            )
        else:
            raise ValueError(f"Unsupported tokenizer: {algorithm}")

    def train(self, texts):
        self.tokenizer.train(texts)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path, algorithm="wordlevel"):
        obj = cls.__new__(cls)
        obj.select_algorithm(algorithm)
        obj.tokenizer.load(path)
        return obj
