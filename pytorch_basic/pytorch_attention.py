
问 Self Attention，并实现 self_attention_layer
# (B, S, H)
class SelfAttention(nn.Moudle):
    def __init__(self):
        pass
    def forward(self, input_seq, lens):
        batch_size, max_seq_len, hidden_dim = input_seq.size()
        input_seq_trans = transpose(input_seq, dim=-1) # B, H, S
        score = torch.matmul(input_seq, input_seq_trans) # B, S, S
        score = softmax(score, dim=1) # B, S, S
        score = reduce_sum(score) # B, S, 1
        weights = mul(score, input_seq) # brocast multiply
        content = reduce_sum(weights, dim=-1)
        return content # B, S, H

