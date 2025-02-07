import torch
from torch import nn, Tensor
from torch.optim import Adam
import random

from transformers import PreTrainedTokenizer
from typing import Literal

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        tokenized_corpus = [tokenizer.tokenize(sentence) for sentence in corpus]
        tokenized_corpus = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_corpus]

        for epoch in range(num_epochs):
            total_loss = 0.0
            for sentence in tokenized_corpus:
                if len(sentence) < 2 * self.window_size + 1:
                    continue
                if self.method == "cbow":
                    loss = self._train_cbow(sentence, criterion, optimizer)
                else:
                    loss = self._train_skipgram(sentence, criterion, optimizer)
                total_loss += loss
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    def _train_cbow(
        self,
        sentence: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        loss = 0
        for i in range(self.window_size, len(sentence) - self.window_size):
            context = sentence[i - self.window_size:i] + sentence[i+1:i+self.window_size+1]
            target = sentence[i]

            context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)
            target_tensor = torch.tensor(target, dtype=torch.long).unsqueeze(0)
            context_embed = self.embeddings(context_tensor).mean(dim=1)
            logits = self.weight(context_embed)
            loss_step = criterion(logits, target_tensor)
            
            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
            loss += loss_step.item()
        return loss

    def _train_skipgram(
        self,
        sentence: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        loss = 0
        for i in range(self.window_size, len(sentence) - self.window_size):
            target = sentence[i]
            context = sentence[i - self.window_size:i] + sentence[i+1:i+self.window_size+1]
            
            target_tensor = torch.tensor(target, dtype=torch.long).to(self.embeddings.weight.device)
            context_tensor = torch.tensor(context, dtype=torch.long).to(self.embeddings.weight.device)
            
            target_embed = self.embeddings(target_tensor)  # (d_model,)
            logits = self.weight(target_embed)  # (vocab_size,)
            
            # 🔹 수정: 여러 context 단어에 대해 반복적으로 예측하도록 변경
            logits = logits.repeat(len(context), 1)  # (context_size, vocab_size)
            
            loss_step = criterion(logits, context_tensor)  # (context_size, vocab_size) vs. (context_size,)
            
            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
            loss += loss_step.item()
        return loss
