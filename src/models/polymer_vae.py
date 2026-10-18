"""
Polymer Variational Autoencoder (PolymerVAE).
Maps discrete polymer SMILES into a continuous latent space R^D and decodes them back autoregressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolymerEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, latent_dim: int = 64):
        super(PolymerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        emb = self.embedding(x) # (B, L, embed_dim)
        _, h_n = self.gru(emb)  # h_n: (2, B, hidden_dim)
        # Concatenate forward and backward final hidden states
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1) # (B, hidden_dim * 2)
        mu = self.fc_mu(h_cat)
        log_var = self.fc_logvar(h_cat)
        return mu, log_var

class PolymerDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, latent_dim: int = 64):
        super(PolymerDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, z):
        # x is target sequence (B, L)
        h_0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0) # (1, B, hidden_dim)
        emb = self.embedding(x)
        out, _ = self.gru(emb, h_0)
        logits = self.fc_out(out) # (B, L, vocab_size)
        return logits

    def generate(self, z, tokenizer, max_len: int = 100, temperature: float = 0.8):
        """Autoregressively decodes a latent vector z into a token sequence."""
        device = z.device
        batch_size = z.size(0)
        h = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0) # (1, B, hidden_dim)

        current_token = torch.full((batch_size, 1), tokenizer.sos_idx, dtype=torch.long, device=device)
        generated_tokens = []

        for _ in range(max_len):
            emb = self.embedding(current_token) # (B, 1, embed_dim)
            out, h = self.gru(emb, h)
            logits = self.fc_out(out.squeeze(1)) / max(temperature, 1e-4) # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            generated_tokens.append(next_token)
            current_token = next_token

        # Stack into (B, max_len)
        return torch.cat(generated_tokens, dim=1)

class PolymerVAE(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, latent_dim: int = 64):
        super(PolymerVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = PolymerEncoder(vocab_size, embed_dim, hidden_dim, latent_dim)
        self.decoder = PolymerDecoder(vocab_size, embed_dim, hidden_dim, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_in, x_target):
        mu, log_var = self.encoder(x_in)
        z = self.reparameterize(mu, log_var)
        logits = self.decoder(x_target, z)
        return logits, mu, log_var, z

    def decode_latent(self, z, tokenizer, max_len: int = 100, temperature: float = 0.8):
        self.eval()
        with torch.no_grad():
            token_tensor = self.decoder.generate(z, tokenizer, max_len=max_len, temperature=temperature)
            results = []
            for row in token_tensor:
                smiles = tokenizer.decode(row)
                results.append(smiles)
            return results

    def sample(self, num_samples: int, tokenizer, device, temperature: float = 0.8):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self.decode_latent(z, tokenizer, max_len=100, temperature=temperature)

def vae_loss_function(logits, targets, mu, log_var, beta: float = 0.05, pad_idx: int = 0):
    """
    Computes Reconstruction Cross-Entropy + beta * KL Divergence.
    """
    # Cross-entropy ignoring pad tokens
    recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_idx)
    # KL divergence
    kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1))
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div
