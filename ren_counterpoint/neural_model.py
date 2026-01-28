import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PolyphonyTransformer(nn.Module):
    """
    Decoder-only transformer for Renaissance polyphony generation.

    Uses causal (autoregressive) attention to predict the next token
    given all previous tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_token_id = pad_token_id
        # self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (improves performance and reduces parameters)
        self.output_proj.weight = self.token_embedding.weight

        # Cached boolean causal mask - no need thanks to flash attention
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer(
            "causal_mask", causal_mask, persistent=False
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
    #     """Generate causal attention mask."""
    #     mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    #     mask = mask.masked_fill(mask == 1, float('-inf'))
    #     return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional padding mask of shape (batch_size, seq_len)
                           1 for real tokens, 0 for padding

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # # 🔴 Slice cached causal mask - not needed with flash attention
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # # 🔴 Proper padding mask (True = ignore)
        # if attention_mask is not None:
        #     key_padding_mask = (attention_mask == 0)
        # else:
        #     key_padding_mask = None

        # Create causal mask
        # causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)#self._generate_causal_mask(seq_len, device)

        # # Create padding mask if provided
        # if attention_mask is not None:
        #     # Convert to key padding mask format (True = ignore)
        #     key_padding_mask = (attention_mask == 0)
        # else:
        #     key_padding_mask = None

        # For decoder-only, we pass the same tensor as both memory and target
        # Using a dummy memory of zeros
        # memory = torch.zeros(batch_size, 1, self.d_model, device=device)

        # === THE FIX: Boolean Mask ===
        # Create a boolean upper-triangular mask.
        # True = Masked (Ignore), False = Attend
        # Size: 8192*8192*1 byte = ~64MB (Safe)
        # causal_mask = torch.triu(
        #     torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        #     diagonal=1
        # )


        # Apply transformer
        x = self.transformer(
            x,
            mask=attn_mask,
            src_key_padding_mask=None,
            is_causal=False,
            # memory,
            # tgt_mask=causal_mask,
            # tgt_key_padding_mask=None,
            # tgt_is_causal=False
        )

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            start_tokens: Initial tokens of shape (batch_size, start_len)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with this threshold
            eos_token_id: If set, stop when this token is generated

        Returns:
            Generated tokens of shape (batch_size, generated_len)
        """
        self.eval()
        device = start_tokens.device
        generated = start_tokens.clone()

        for _ in range(max_length - start_tokens.size(1)):
            # Get logits for last position
            logits = self(generated)[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
