from .neural_model import PolyphonyTransformer
from .neurosymbolic_model import Neurosymbolic Model
from .tokenizer import RemiPlusTokenizer
import os

def load_neurosymbolic_model(checkpoint_dir, tokenizer, checkpoint_name: str = 'best_model.pt', device='cuda'):
    path = os.path.join(checkpoint_dir, checkpoint_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No best_model.pt found in {checkpoint_dir}")

    checkpoint = torch.load(path, map_location=device)

    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif 'config' in checkpoint:
        config = checkpoint['config']
    elif 'args' in checkpoint:
        config = checkpoint['args']
    else:
        raise KeyError(f"Could not find configuration in {path}. Keys found: {checkpoint.keys()}")

    try:
        raw_model = PolyphonyTransformer(**config)
    except TypeError as e:
        import inspect
        valid_args = inspect.signature(PolyphonyTransformer.__init__).parameters
        filtered_config = {k: v for k, v in config.items() if k in valid_args and k != 'self'}
        raw_model = PolyphonyTransformer(**filtered_config)

    raw_model.load_state_dict(checkpoint['model_state_dict'])

    raw_model.to(device)
    raw_model.eval()

    print(f"Loaded model from {path}")
    return NeurosymbolicModel(raw_model, tokenizer)



def generate_sequence(
    model: NeurosymbolicModel,
    bos_id: int,
    eos_id: int,
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Generate a sequence using the Neurosymbolic wrapper."""

    # Note: We do NOT call model.to(device) here because NeurosymbolicModel
    # is a wrapper, not an nn.Module. The internal model should already be
    # on the correct device from the loading step.

    print(f"\nGenerating with NeurosymbolicModel:")
    print(f"  Max length: {max_length}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")
    print(f"  Device: {device}")

    # Start with BOS token
    start_tokens = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    # We don't strictly need 'with torch.no_grad()' here because
    # your generate method has the decorator, but it's good practice.
    generated = model.generate(
        start_tokens=start_tokens,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_id
    )

    # Convert to list
    token_ids = generated[0].cpu().tolist()

    print(f"\nGeneration complete!")
    print(f"  Total tokens: {len(token_ids)}")
    print(f"  Started with BOS: {token_ids[0] == bos_id}")
    print(f"  Ended with EOS: {token_ids[-1] == eos_id}")

    return token_ids



