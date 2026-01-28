from collections import defaultdict, Counter
from dataclasses import dataclass, field
import mido

"""
Comprehensive MIDI Tokenization Verification

Verifies that REMI+ tokenization preserves all notes with their correct program
assignments by comparing:
- Ground truth: mido (reliable program parsing)
- Token sequence: actual Program tokens in the REMI+ output

This does NOT use symusic for verification since symusic has bugs in program parsing.

Usage:
    from verify_tokenization import verify_all_files, print_report, diagnose_file

    # Check all files with detailed report
    report = verify_all_files(tokenizer, midi_paths)
    print_report(report)

    # Diagnose specific problematic files
    diagnose_file(tokenizer, "path/to/problem.mid")
"""

from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import mido


@dataclass
class Note:
    """A note with all attributes needed for comparison."""
    pitch: int
    start_tick: int
    duration: int
    program: int

    def __repr__(self):
        return f"Note(p={self.pitch}, t={self.start_tick}, dur={self.duration}, prog={self.program})"


@dataclass
class VerificationResult:
    """Detailed results of tokenization verification."""
    path: str
    success: bool = True

    # Stats
    original_note_count: int = 0
    token_note_count: int = 0
    original_programs: set = field(default_factory=set)
    token_programs: set = field(default_factory=set)

    # Per-program counts
    original_notes_by_program: dict = field(default_factory=dict)
    token_notes_by_program: dict = field(default_factory=dict)

    # Errors
    missing_notes: list = field(default_factory=list)
    extra_notes: list = field(default_factory=list)
    wrong_program_notes: list = field(default_factory=list)  # (original_note, token_program)
    pitch_mismatches: list = field(default_factory=list)  # Details of pitch-level mismatches

    error_message: str = ""

    def __str__(self):
        if self.success:
            return f"✓ {self.path}: OK ({self.original_note_count} notes, programs {self.original_programs})"

        lines = [f"✗ {self.path}: FAILED"]

        if self.original_programs != self.token_programs:
            lines.append(f"  Programs: {self.original_programs} -> {self.token_programs}")

        if self.original_note_count != self.token_note_count:
            lines.append(f"  Note count: {self.original_note_count} -> {self.token_note_count}")

        # Per-program breakdown
        all_progs = self.original_programs | self.token_programs
        for prog in sorted(all_progs):
            orig = self.original_notes_by_program.get(prog, 0)
            tok = self.token_notes_by_program.get(prog, 0)
            if orig != tok:
                lines.append(f"    Program {prog}: {orig} -> {tok}")

        if self.pitch_mismatches:
            lines.append(f"  Pitch-level mismatches: {len(self.pitch_mismatches)}")
            for mismatch in self.pitch_mismatches[:5]:
                lines.append(f"    Pitch {mismatch['pitch']}: mido={mismatch['mido_progs']}, tokens={mismatch['token_progs']}")
            if len(self.pitch_mismatches) > 5:
                lines.append(f"    ... and {len(self.pitch_mismatches) - 5} more")

        if self.wrong_program_notes:
            lines.append(f"  Wrong program: {len(self.wrong_program_notes)} notes")
            for orig_note, tok_prog in self.wrong_program_notes[:5]:
                lines.append(f"    {orig_note} -> assigned to program {tok_prog}")
            if len(self.wrong_program_notes) > 5:
                lines.append(f"    ... and {len(self.wrong_program_notes) - 5} more")

        if self.missing_notes:
            lines.append(f"  Missing notes: {len(self.missing_notes)}")
            for note in self.missing_notes[:3]:
                lines.append(f"    {note}")
            if len(self.missing_notes) > 3:
                lines.append(f"    ... and {len(self.missing_notes) - 3} more")

        if self.extra_notes:
            lines.append(f"  Extra notes: {len(self.extra_notes)}")

        if self.error_message:
            lines.append(f"  Error: {self.error_message}")

        return "\n".join(lines)


def extract_notes_from_mido(midi_path: str | Path) -> list[Note]:
    """
    Extract all notes from MIDI using mido (ground truth).

    Returns list of Note objects with correct program assignments.
    """
    mid = mido.MidiFile(midi_path)
    notes = []

    for track in mid.tracks:
        # Find program for this track
        program = 0  # Default
        for msg in track:
            if msg.type == 'program_change':
                program = msg.program
                break

        # Parse notes
        active_notes = {}  # (channel, pitch) -> start_tick
        current_tick = 0

        for msg in track:
            current_tick += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[(msg.channel, msg.note)] = current_tick
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes:
                    start_tick = active_notes.pop(key)
                    duration = current_tick - start_tick
                    notes.append(Note(
                        pitch=msg.note,
                        start_tick=start_tick,
                        duration=duration,
                        program=program
                    ))

    return notes


def extract_program_pitch_pairs_from_tokens(tokenizer, tokens) -> list[tuple[int, int]]:
    """
    Extract (program, pitch) pairs from token sequence in order.

    This is the most reliable way to verify program assignments since it
    doesn't depend on timing reconstruction.
    """
    if hasattr(tokens, 'ids'):
        token_ids = tokens.ids
    else:
        token_ids = tokens

    events = tokenizer.tokenizer._ids_to_tokens(token_ids)

    pairs = []
    current_program = 0

    for event in events:
        if event.startswith("Program_"):
            current_program = int(event.split("_")[1])
        elif event.startswith("Pitch_"):
            pitch = int(event.split("_")[1])
            pairs.append((current_program, pitch))

    return pairs


def compare_multisets(list1: list, list2: list) -> bool:
    """Compare two lists as multisets (order doesn't matter, counts do)."""
    return Counter(list1) == Counter(list2)


def verify_tokenization(tokenizer, midi_path: str | Path, verbose: bool = False) -> VerificationResult:
    """
    Verify that every note in the token sequence has the correct program.

    Compares mido ground truth against actual token sequence.

    The verification checks:
    1. Same set of programs
    2. Same total note count
    3. Same note count per program
    4. For each pitch, same multiset of program assignments
    """
    result = VerificationResult(path=str(midi_path))

    try:
        # Get ground truth from mido
        original_notes = extract_notes_from_mido(midi_path)
        original_notes.sort(key=lambda n: (n.start_tick, n.pitch, n.program))

        result.original_note_count = len(original_notes)
        result.original_programs = {n.program for n in original_notes}
        result.original_notes_by_program = dict(Counter(n.program for n in original_notes))

        # Tokenize
        tokens = tokenizer(midi_path)

        # Extract (program, pitch) pairs from tokens
        token_pairs = extract_program_pitch_pairs_from_tokens(tokenizer, tokens)

        result.token_note_count = len(token_pairs)
        result.token_programs = {p for p, _ in token_pairs}
        result.token_notes_by_program = dict(Counter(p for p, _ in token_pairs))

        if verbose:
            print(f"Original: {result.original_note_count} notes, programs {result.original_programs}")
            print(f"  Per program: {result.original_notes_by_program}")
            print(f"Tokens: {result.token_note_count} notes, programs {result.token_programs}")
            print(f"  Per program: {result.token_notes_by_program}")

        # === CHECK 1: Program sets match ===
        if result.original_programs != result.token_programs:
            result.success = False
            if verbose:
                print(f"FAIL: Program sets don't match")

        # === CHECK 2: Total note counts match ===
        if result.original_note_count != result.token_note_count:
            result.success = False
            if verbose:
                print(f"FAIL: Note counts don't match")

        # === CHECK 3: Per-program counts match ===
        if result.original_notes_by_program != result.token_notes_by_program:
            result.success = False
            if verbose:
                print(f"FAIL: Per-program counts don't match")

        # === CHECK 4: For each pitch, program assignments match as multisets ===
        # This catches cases where notes got assigned to wrong programs even if totals match

        # Group mido notes by pitch
        mido_by_pitch = defaultdict(list)
        for n in original_notes:
            mido_by_pitch[n.pitch].append(n.program)

        # Group token notes by pitch
        token_by_pitch = defaultdict(list)
        for prog, pitch in token_pairs:
            token_by_pitch[pitch].append(prog)

        all_pitches = set(mido_by_pitch.keys()) | set(token_by_pitch.keys())

        for pitch in all_pitches:
            mido_progs = mido_by_pitch.get(pitch, [])
            token_progs = token_by_pitch.get(pitch, [])

            # Compare as multisets using Counter
            if Counter(mido_progs) != Counter(token_progs):
                result.success = False
                result.pitch_mismatches.append({
                    'pitch': pitch,
                    'mido_progs': sorted(mido_progs),
                    'token_progs': sorted(token_progs),
                })
                if verbose:
                    print(f"FAIL: Pitch {pitch} mismatch: mido={sorted(mido_progs)}, tokens={sorted(token_progs)}")

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        if verbose:
            import traceback
            traceback.print_exc()

    return result


def verify_all_files(tokenizer, midi_paths: list, verbose: bool = False,
                     show_progress: bool = True) -> dict:
    """
    Verify all MIDI files and return comprehensive report.
    """
    results = []

    if show_progress:
        try:
            from tqdm import tqdm
            paths_iter = tqdm(midi_paths, desc="Verifying")
        except ImportError:
            paths_iter = midi_paths
            print(f"Verifying {len(midi_paths)} files...")
    else:
        paths_iter = midi_paths

    for path in paths_iter:
        result = verify_tokenization(tokenizer, path, verbose=verbose)
        results.append(result)

    # Compile summary
    failed = [r for r in results if not r.success]

    summary = {
        'total_files': len(results),
        'passed': len(results) - len(failed),
        'failed': len(failed),
        'pass_rate': (len(results) - len(failed)) / len(results) * 100 if results else 0,
    }

    # Categorize errors
    error_types = defaultdict(list)
    for r in failed:
        if r.error_message:
            error_types['exceptions'].append(r.path)
        if r.original_programs != r.token_programs:
            error_types['program_set_mismatch'].append(r.path)
        if r.original_note_count != r.token_note_count:
            error_types['note_count_mismatch'].append(r.path)
        if r.original_notes_by_program != r.token_notes_by_program:
            error_types['per_program_count_mismatch'].append(r.path)
        if r.pitch_mismatches:
            error_types['pitch_program_mismatch'].append(r.path)

    return {
        'results': results,
        'summary': summary,
        'failed_files': [r.path for r in failed],
        'failed_results': failed,
        'error_types': dict(error_types),
    }


def print_report(report: dict):
    """Print formatted verification report."""
    summary = report['summary']

    print("\n" + "=" * 70)
    print("TOKENIZATION VERIFICATION REPORT")
    print("=" * 70)
    print(f"\nTotal files:  {summary['total_files']}")
    print(f"Passed:       {summary['passed']}")
    print(f"Failed:       {summary['failed']}")
    print(f"Pass rate:    {summary['pass_rate']:.1f}%")

    if report['error_types']:
        print("\n" + "-" * 70)
        print("ERROR CATEGORIES:")
        print("-" * 70)
        for error_type, files in sorted(report['error_types'].items()):
            print(f"  {error_type}: {len(files)} files")

    if report['failed_results']:
        print("\n" + "-" * 70)
        print("FAILED FILES:")
        print("-" * 70)
        for result in report['failed_results']:
            print(f"\n{result}")

    print("\n" + "=" * 70)


def diagnose_file(tokenizer, midi_path: str | Path):
    """
    Detailed diagnosis of a problematic file.

    Shows exactly what mido sees vs what ends up in the tokens,
    using the SAME comparison logic as verify_tokenization.
    """
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {midi_path}")
    print('='*70)

    # === MIDO GROUND TRUTH ===
    print(f"\n--- MIDO GROUND TRUTH ---")
    mid = mido.MidiFile(midi_path)

    mido_notes = extract_notes_from_mido(midi_path)

    # Show per-track info
    for i, track in enumerate(mid.tracks):
        program = None
        note_count = 0
        pitches = []

        for msg in track:
            if msg.type == 'program_change':
                program = msg.program
            if msg.type == 'note_on' and msg.velocity > 0:
                note_count += 1
                pitches.append(msg.note)

        if note_count > 0:
            print(f"  Track {i}: program={program}, notes={note_count}, "
                  f"pitch_range=[{min(pitches)}-{max(pitches)}]")

    # Per-program summary
    mido_by_program = defaultdict(list)
    for n in mido_notes:
        mido_by_program[n.program].append(n.pitch)

    print(f"\n  Notes per program (mido):")
    for prog in sorted(mido_by_program.keys()):
        pitches = mido_by_program[prog]
        print(f"    Program {prog}: {len(pitches)} notes, pitches {min(pitches)}-{max(pitches)}")

    # === TOKEN SEQUENCE ===
    print(f"\n--- TOKEN SEQUENCE ---")
    tokens = tokenizer(midi_path)
    events = tokenizer.tokenizer._ids_to_tokens(tokens.ids)

    # Count program tokens
    program_tokens = [e for e in events if e.startswith("Program_")]
    print(f"  Total tokens: {len(tokens.ids)}")
    print(f"  Program tokens: {len(program_tokens)}")
    print(f"  Unique programs in tokens: {set(program_tokens)}")

    # Extract (program, pitch) pairs from tokens
    token_pairs = extract_program_pitch_pairs_from_tokens(tokenizer, tokens)

    token_by_program = defaultdict(list)
    for prog, pitch in token_pairs:
        token_by_program[prog].append(pitch)

    print(f"\n  Notes per program (tokens):")
    for prog in sorted(token_by_program.keys()):
        pitches = token_by_program[prog]
        print(f"    Program {prog}: {len(pitches)} notes, pitches {min(pitches)}-{max(pitches)}")

    # === COMPARISON BY PROGRAM ===
    print(f"\n--- COMPARISON BY PROGRAM ---")
    all_programs = set(mido_by_program.keys()) | set(token_by_program.keys())

    program_ok = True
    for prog in sorted(all_programs):
        mido_pitches = sorted(mido_by_program.get(prog, []))
        token_pitches = sorted(token_by_program.get(prog, []))

        if mido_pitches == token_pitches:
            print(f"  Program {prog}: ✓ MATCH ({len(mido_pitches)} notes)")
        else:
            program_ok = False
            print(f"  Program {prog}: ✗ MISMATCH")
            print(f"    Mido:   {len(mido_pitches)} notes")
            print(f"    Tokens: {len(token_pitches)} notes")

            mido_counter = Counter(mido_pitches)
            token_counter = Counter(token_pitches)

            for pitch in sorted(set(mido_pitches) | set(token_pitches)):
                mc = mido_counter.get(pitch, 0)
                tc = token_counter.get(pitch, 0)
                if mc != tc:
                    print(f"      Pitch {pitch}: {mc} in mido, {tc} in tokens")

    # === COMPARISON BY PITCH (the actual verification logic) ===
    print(f"\n--- COMPARISON BY PITCH ---")

    mido_by_pitch = defaultdict(list)
    for n in mido_notes:
        mido_by_pitch[n.pitch].append(n.program)

    token_by_pitch = defaultdict(list)
    for prog, pitch in token_pairs:
        token_by_pitch[pitch].append(prog)

    all_pitches = set(mido_by_pitch.keys()) | set(token_by_pitch.keys())

    pitch_mismatches = []
    for pitch in sorted(all_pitches):
        mido_progs = mido_by_pitch.get(pitch, [])
        token_progs = token_by_pitch.get(pitch, [])

        if Counter(mido_progs) != Counter(token_progs):
            pitch_mismatches.append((pitch, mido_progs, token_progs))

    if pitch_mismatches:
        print(f"  Found {len(pitch_mismatches)} pitch(es) with program mismatches:")
        for pitch, mido_progs, token_progs in pitch_mismatches[:10]:
            print(f"    Pitch {pitch}: mido={sorted(mido_progs)}, tokens={sorted(token_progs)}")
        if len(pitch_mismatches) > 10:
            print(f"    ... and {len(pitch_mismatches) - 10} more")
    else:
        print(f"  ✓ All pitches have correct program assignments!")

    # === FINAL VERDICT ===
    print(f"\n--- VERDICT ---")
    result = verify_tokenization(tokenizer, midi_path)
    if result.success:
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL")
        if result.original_programs != result.token_programs:
            print(f"    - Program set mismatch")
        if result.original_note_count != result.token_note_count:
            print(f"    - Note count mismatch: {result.original_note_count} vs {result.token_note_count}")
        if result.original_notes_by_program != result.token_notes_by_program:
            print(f"    - Per-program count mismatch")
        if result.pitch_mismatches:
            print(f"    - {len(result.pitch_mismatches)} pitch-level program mismatches")

    # === Show token sequence around first program token ===
    print(f"\n--- FIRST 50 MUSICAL TOKENS ---")
    musical_tokens = [e for e in events if not e.startswith(("PAD", "BOS", "EOS"))]
    for i, event in enumerate(musical_tokens[:50]):
        print(f"  {i:3d}: {event}")

    print('='*70 + "\n")

    return result


def debug_single_file(tokenizer, midi_path: str | Path):
    """
    Run verification with verbose output to see exactly what's being compared.
    """
    print(f"Debugging: {midi_path}")
    print("-" * 50)
    return verify_tokenization(tokenizer, midi_path, verbose=True)

