import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        self.hidden_size = hidden_size
        
        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size)
        
        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))  # Update gate
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))  # Reset gate
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h))  # Candidate hidden state
        h_next = (1 - z) * h + z * h_tilde  # New hidden state
        return h_next


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        batch_size, seq_len, _ = inputs.shape
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)  # 초기 hidden state
        
        for t in range(seq_len):
            h = self.cell(inputs[:, t, :], h)
        
        return h  # 최종 hidden state 반환