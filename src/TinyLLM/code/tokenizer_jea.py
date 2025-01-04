import os
from typing import List

from sentencepiece import SentencePieceProcessor

DEFAULT_TOKEN_MODEL = "./data_jea/token/token4096.model"


class Tokenizer(object):
    def __init__(self, tokenizer_model=None):
        tokenizer_model_path = tokenizer_model if tokenizer_model is None else DEFAULT_TOKEN_MODEL
        assert os.path.isfile(tokenizer_model_path), tokenizer_model_path

        # 确认完成后，根据路径加载分词器模型
        self.sp_token_model = SentencePieceProcessor.Init(model_file=tokenizer_model_path)
        self.model_path = tokenizer_model_path

        # 获取分词器的特殊token和词汇表大小
        self.n_words: int = self.sp_token_model.vocab_size()  # 词汇表大小
        self.bos_id: int = self.sp_token_model.bos_id()  # 句子开头 (BOS) 的ID
        self.eos_id: int = self.sp_token_model.eos_id()  # 句子结尾 (EOS) 的ID
        self.pad_id: int = self.sp_token_model.pad_id()  # 填充 (PAD) 的ID

        # 验证分词器词汇表大小是否正确
        assert self.sp_token_model.vocab_size() == self.sp_token_model.get_vocab_size()
        pass

    def encode(self, text: str, bos: bool, eos: bool):
        assert type(text) is str

        tokens = self.sp_token_model.encode(text)
        if bos:
            tokens = [self.bos_id] + tokens
        # 如果需要EOS标记，将其添加到词元列表末尾
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.sp_token_model.decode(tokens)
        pass
