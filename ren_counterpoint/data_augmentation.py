import mido
from pathlib import Path
import multiprocessing
from functools import partial
from tqdm.auto import tqdm

def transpose_midi(midi_file: mido.MidiFile, semitones: int) -> mido.MidiFile:
    """
    Transpose all note events in a MIDI file by the given number of semitones.

    Args:
        midi_file: The source MIDI file
        semitones: Number of semitones to transpose (positive = up, negative = down)

    Returns:
        A new MidiFile with transposed notes
    """
    # Create a new MIDI file with the same settings
    new_midi = mido.MidiFile(ticks_per_beat=midi_file.ticks_per_beat, type=midi_file.type)
    for track in midi_file.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type in ('note_on', 'note_off') and msg.note != 0:
                # Transpose the note, clamping to valid MIDI range (0-127)
                new_note = msg.note + semitones
                assert 0 <= new_note <= 127, f"MIDI file has transposed note of pitch {new_note}, out of range"
                new_msg = msg.copy(note=new_note)
                new_track.append(new_msg)
            else:
                new_track.append(msg.copy())

        new_midi.tracks.append(new_track)
    return new_midi


def augment_midi_file(input_path: Path, output_dir: Path, min_semitones: int = -5, max_semitones: int = 6, verbose = False):
    """
    Create transposed versions of a single MIDI file.

    Args:
        input_path: Path to the input MIDI file
        output_dir: Directory to save transposed files
        min_semitones: Minimum transposition (default: -5, down 5 semitones)
        max_semitones: Maximum transposition (default: +6, up 6 semitones)
    """
    try:
        midi_file = mido.MidiFile(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return 0

    base_name = input_path.stem
    extension = input_path.suffix
    assert extension == ".mid", f"{input_path} does not end with .mid"

    count = 0
    for semitones in range(min_semitones, max_semitones + 1):
        # suffix = format_transposition_suffix(semitones)
        if semitones >= 0:
          output_name = f"{base_name}_+{abs(semitones):02d}{extension}"
        else:
          output_name = f"{base_name}_-{abs(semitones):02d}{extension}"
        output_path = output_dir / Path(output_name)

        transposed = transpose_midi(midi_file, semitones)
        transposed.save(output_path)
        count += 1
        if verbose:
          print(f"  Created: {output_name}")
    return count

def worker_augment_file(input_path: Path, output_dir: Path, min_semitones: int, max_semitones: int) -> tuple[Path, int]:
    """
    Worker function for multiprocessing.

    Returns:
        Tuple of (input_path, count) for reporting
    """
    count = augment_midi_file(input_path, output_dir, min_semitones, max_semitones, verbose=False)
    return (input_path, count)

def process_directory_parallel(
    midi_paths: list[str | Path],
    output_dir: str | Path,
    min_semitones: int = -5,
    max_semitones: int = 6,
    num_workers: int = None,
    verbose = False,
) -> int:
    """
    Process all MIDI files in a directory using multiple processes.

    Args:
        input_dir: Directory containing MIDI files
        output_dir: Directory to save transposed files
        min_semitones: Minimum transposition
        max_semitones: Maximum transposition
        num_workers: Number of worker processes (default: number of CPUs)

    Returns:
        Total number of files created
    """
    assert midi_paths, f"List of midi_paths: {midi_paths}"

    midi_paths = [Path(p) for p in midi_paths]
    output_dir = Path(output_dir)

    # Default to number of CPUs
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    assert max_semitones > min_semitones, f"max_semitones={max_semitones}, min_semitones={min_semitones}"
    num_versions = max_semitones - min_semitones + 1
    print(f"Transposition range: {min_semitones} to {max_semitones} ({num_versions} versions each)")
    print(f"Output directory: {output_dir}")
    print(f"Using {num_workers} workers\n")

    worker_fn = partial(
        worker_augment_file,
        output_dir=output_dir,
        min_semitones=min_semitones,
        max_semitones=max_semitones
    )


    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
                       pool.imap(worker_fn, midi_paths),
                       total=len(midi_paths)))#pool.map(worker_fn, [Path(midi) for midi in midi_paths])

    total_created = 0
    for input_path, count in results:
        if verbose:
          print(f"  {input_path.name}: {count} files")
        if count < num_versions:
          print(f"  {input_path} only produced {count} versions, should be {num_versions}")
        total_created += count

    print(f"\nTotal files created: {total_created}")
    return total_created

