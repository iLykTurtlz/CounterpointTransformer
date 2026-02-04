import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .neural_model import PolyphonyTransformer
from .symbolic_model import REMIConstraints, CounterpointSolver, REMIState
from .tokenizer import RemiPlusTokenizer
from tqdm.auto import tqdm

class NeurosymbolicModel:
    def __init__(
        self,
        neural_model: nn.Module,
        tokenizer: RemiPlusTokenizer,
        use_counterpoint_solver: bool = True,
    ):
        self.neural_model = neural_model
        self.tokenizer = tokenizer
        self.use_counterpoint_solver = use_counterpoint_solver
        self.name_to_id = self.tokenizer.tokenizer.vocab
        self.id_to_name = {v:k for k,v in self.name_to_id.items()}
        self.bnf_constrainer = REMIConstraints(self.name_to_id, self.id_to_name)
        self.counterpoint_solver = CounterpointSolver(self.name_to_id, self.id_to_name)
        self.vocab_size = len(self.name_to_id)



    def get_domains(self, state):
        """Invoke the BNF and Z3 models to get valid token domains"""
        domains = self.bnf_constrainer.get_valid_domains(state)
        pitch_domain = None
        idx = None
        for i,domain in enumerate(domains):
            if domain.name == 'Pitch':
                pitch_domain = domain
                idx = i 
                assert all(d.name != 'Pitch' for d in domains[(i+1):])
                break
        if pitch_domain is not None:
            forbidden = self.counterpoint_solver.solve_pitch_constraints(state, pitch_domain)
            domains[idx] = TokenSet(
                {tok for tok in range(pitch_domain.start, pitch_domain.end) if tok not in forbidden},
                'Pitch')
        return domains

    def get_invalid_mask(self, state, device) -> torch.Tensor:
        """
        Create boolean mask: True for INVALID tokens

        Args:
            state: Current generation state

        Returns:
            Boolean array of shape [vocab_size] where True = invalid
        """
        # Start with all tokens invalid
        mask = torch.ones(self.vocab_size, dtype=torch.bool)

        # Mark valid domains as False
        valid = self.get_domains(state)
        for domain in valid:
            if isinstance(domain, TokenRange):
                mask[ domain.start : domain.end ] = False
            elif isinstance(domain, TokenSet):
                mask[ list(domain.ids) ] = False
            else:
                raise ValueError(f"Unknown domain instance: {domain}")
        # PitchDrum is already excluded - nothing to do
        return mask.to(device)

    def mask_logits(self, state, logits):
        mask = self.get_invalid_mask(state, logits.device)
        mask = mask.unsqueeze(0)
        mask = mask.expand_as(logits)
        logits[mask] = float('-inf')
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
    ):
        self.neural_model.eval()
        device = start_tokens.device

        state = REMIState()
        cpu_tokens = start_tokens[0].tolist()
        for token in cpu_tokens:
            state.update(token, self.id_to_name[token])


        generated = start_tokens.clone()
        start_len = start_tokens.size(1)
        for _ in tqdm(range(max_length - start_len)):
            logits = self.neural_model(generated)[:, -1, :] / temperature
            logits = self.mask_logits(state, logits)
            if top_k is not None:
                # indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                # logits[indices_to_remove] = float('-inf')
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[..., [-1]]] = float('-inf')
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_int = next_token.item()
            state.update(next_token_int, self.id_to_name[next_token_int])

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token_int == eos_token_id):
                break
        return generated

    def load_best_model(self, checkpoint_dir, device='cuda'):
        """
        Loads 'best_model.pt' from the specified directory.
        """
        path = os.path.join(checkpoint_dir, 'best_model.pt')
    
        if not os.path.exists(path):
            raise FileNotFoundError(f"No best_model.pt found in {checkpoint_dir}")
    
        # Load the file
        # map_location ensures we don't crash if the model was saved on a different GPU index
        checkpoint = torch.load(path, map_location=device)
    
        # Handle both full training checkpoints (dict) and direct weight saves
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.neural_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.neural_model.load_state_dict(checkpoint)
    
        # Ensure we are in eval mode immediately after loading
        self.neural_model.to(device)
        self.neural_model.eval()
    
        print(f"Successfully loaded best model from {path}")
    
