import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
import multiprocessing
from functools import partial
from tqdm.auto import tqdm
from pathlib import Path

class PolyphonyTorchDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for HuggingFace Dataset.

    Handles:
    - Random cropping for long sequences
    - BOS/EOS token insertion
    - Padding
    """

    def __init__(
        self,
        hf_dataset: HFDataset,
        max_seq_len: int = 1024,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
    ):
        self.dataset = hf_dataset
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        token_ids = item["token_ids"]
        seq_len = len(token_ids)

        # Determine crop boundaries
        if seq_len + 2 <= self.max_seq_len:
            # Whole sequence fits
            start_idx, end_idx = 0, seq_len
        else:
            # Random crop (reserve space for up to 2 special tokens)
            max_content = self.max_seq_len - 2
            max_start = seq_len - max_content
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + max_content

        # Determine special tokens
        has_bos = (start_idx == 0)
        has_eos = (end_idx == seq_len)

        # Build sequence
        content = token_ids[start_idx:end_idx]

        sequence = []
        if has_bos:
            sequence.append(self.bos_id)
        sequence.extend(content)
        if has_eos:
            sequence.append(self.eos_id)

        # Ensure the constructed sequence isn't too long
        assert(len(sequence) <= self.max_seq_len)

        # Create attention mask and pad
        attention_mask = [1] * len(sequence)
        padding_length = self.max_seq_len - len(sequence)

        if padding_length > 0:
            sequence = sequence + [self.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

def tokenize_single_file(path: str, tokenizer_path: str) -> dict:
    """
    Tokenize a single MIDI file.

    This function is designed to be called in parallel.
    It loads the tokenizer fresh in each process to avoid pickling issues.
    """
    try:
        tokenizer = RemiPlusTokenizer(load_path=tokenizer_path)
        vocab_size = len(tokenizer.tokenizer)

        # Tokenize
        tok_sequence = tokenizer(path)

        # Extract token IDs
        if hasattr(tok_sequence, 'ids'):
            token_ids = tok_sequence.ids
        else:
            token_ids = list(tok_sequence)

        # Handle nested lists
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        # Convert to Python ints and validate
        token_ids = [int(t) for t in token_ids]

        if any(t < 0 or t >= vocab_size for t in token_ids):
            return {"token_ids": [], "length": 0, "error": "invalid_tokens"}

        return {
            "token_ids": token_ids,
            "length": len(token_ids),
            "error": None,
        }

    except Exception as e:
        return {"token_ids": [], "length": 0, "error": str(e)}


def build_polyphony_dataset(
    midi_paths: list[str],
    tokenizer_path: str,
    output_path: str | None = None,
    num_workers: int | None = None,
    chunk_size: int = 100,
) -> HFDataset:
    """
    Build a HuggingFace Dataset from MIDI files with parallel processing.

    Args:
        midi_paths: List of MIDI file paths
        tokenizer_path: Path to saved MidiTok tokenizer
        output_path: Where to save the dataset (optional)
        num_workers: Number of parallel workers (default: CPU count)
        chunk_size: Number of files per chunk for progress reporting

    Returns:
        HuggingFace Dataset
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"Tokenizing {len(midi_paths)} MIDI files using {num_workers} workers...")

    # Create partial function with tokenizer path
    tokenize_fn = partial(tokenize_single_file, tokenizer_path=tokenizer_path)

    # Process in parallel with progress bar
    results = []
    filenames = []

    with multiprocessing.Pool(num_workers) as pool:
        with tqdm(total=len(midi_paths), desc="Tokenizing") as pbar:
            for i, result in enumerate(pool.imap(tokenize_fn, midi_paths, chunksize=chunk_size)):
                results.append(result)
                filenames.append(Path(midi_paths[i]).stem)
                pbar.update(1)

    # Separate successful and failed
    successful = []
    failed = []

    for filename, result in zip(filenames, results):
        if result["error"] is None and len(result["token_ids"]) > 0:
            successful.append({
                "filename": filename,
                "token_ids": result["token_ids"],
                "length": result["length"],
            })
        else:
            failed.append({"filename": filename, "error": result["error"]})

    print(f"\nTokenization complete:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if failed and len(failed) <= 10:
        print(f"  Failed files: {[f['filename'] for f in failed]}")
    elif failed:
        print(f"  First 10 failed: {[f['filename'] for f in failed[:10]]}")

    # Create HuggingFace Dataset
    dataset = HFDataset.from_list(successful)

    # Print statistics
    lengths = dataset["length"]
    print(f"\nHFDataset statistics:")
    print(f"  Total sequences: {len(dataset)}")
    print(f"  Total tokens: {sum(lengths):,}")
    print(f"  Min length: {min(lengths)}")
    print(f"  Max length: {max(lengths)}")
    print(f"  Mean length: {sum(lengths) / len(lengths):.1f}")

    # Save if path provided
    if output_path:
        dataset.save_to_disk(output_path)
        print(f"\nHFDataset saved to: {output_path}")

    return dataset
