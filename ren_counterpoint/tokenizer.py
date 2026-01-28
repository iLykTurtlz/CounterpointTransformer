from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score

class RemiPlusTokenizer:
    """
    A REMI+ tokenizer configured for Renaissance polyphony.

    REMI+ is obtained by using REMI with:
    - use_programs=True
    - one_token_stream_for_programs=True
    - use_time_signatures=True
    """

    # Supported time signatures (numerator, denominator) tuples
    SUPPORTED_TIME_SIGNATURES = [
        (3, 8), (12, 8), (6, 8),  # compound meters
        (5, 4), (6, 4), (3, 4), (2, 4), (1, 4), (4, 4)  # simple meters
    ]

    # Default configuration values
    DEFAULT_PITCH_RANGE = (21, 109)
    DEFAULT_BEAT_RES = {
        (0, 4): 24,    # High precision for short notes (triplets/16ths)
        (4, 12): 4,    # 16th note precision for medium notes
        (12, 100): 1,  # 1 beat precision for long pedal points
    }
    DEFAULT_NUM_VELOCITIES = 1
    DEFAULT_SPECIAL_TOKENS = ["PAD", "BOS", "EOS"]
    DEFAULT_PROGRAMS = list(range(0, 12))

    def __init__(
        self,
        pitch_range: tuple[int, int] | None = None,
        beat_res: dict | None = None,
        num_velocities: int | None = None,
        special_tokens: list[str] | None = None,
        programs: list[int] | None = None,
        load_path: str | None = None,
    ):
        """
        Initialize the REMI+ tokenizer.

        Args:
            pitch_range: Tuple of (min_pitch, max_pitch). Defaults to piano range (21, 109).
            beat_res: Beat resolution dict mapping beat ranges to samples per beat.
            num_velocities: Number of velocity bins. Defaults to 1 (no dynamics).
            special_tokens: List of special tokens. Defaults to ["PAD", "BOS", "EOS"].
            programs: List of MIDI programs to use. Defaults to 0-11.
        """
        if load_path:
          self._tokenizer = REMI(params=load_path)
        else:
          self._config = self._create_config(
              pitch_range=pitch_range or self.DEFAULT_PITCH_RANGE,
              beat_res=beat_res or self.DEFAULT_BEAT_RES,
              num_velocities=num_velocities or self.DEFAULT_NUM_VELOCITIES,
              special_tokens=special_tokens or self.DEFAULT_SPECIAL_TOKENS,
              programs=programs or self.DEFAULT_PROGRAMS,
          )
          self._tokenizer = REMI(self._config)

    def _create_config(
        self,
        pitch_range: tuple[int, int],
        beat_res: dict,
        num_velocities: int,
        special_tokens: list[str],
        programs: list[int],
    ) -> TokenizerConfig:
        """Create the TokenizerConfig for REMI+."""
        return TokenizerConfig(
            pitch_range=pitch_range,
            beat_res=beat_res,
            # num_velocities=num_velocities,
            special_tokens=special_tokens,

            use_velocities=False,
            # === REMI+ specific settings ===
            use_time_signatures=True,
            use_programs=True,
            one_token_stream_for_programs=True,

            # === Disabled features (not relevant for Renaissance music) ===
            use_chords=False,
            use_tempos=False,
            use_rests=False,
            use_sustain_pedals=False,
            use_pitch_bends=False,
            use_pitch_intervals=False,

            # Time signature handling
            delete_equal_successive_time_sig_changes=True,

            programs=programs,
        )

    def train_bpe(self, files_paths: list[Path], vocab_size: int):
      self._tokenizer.train(
          vocab_size=vocab_size,
          model="BPE",
          files_paths=files_paths,
      )

    @property
    def tokenizer(self) -> REMI:
        """Access the underlying REMI tokenizer."""
        return self._tokenizer

    @property
    def config(self) -> TokenizerConfig:
        """Access the tokenizer configuration."""
        return self._config


    def decode(self, tokens):
        """
        Decode tokens back to a MIDI file.

        Args:
            tokens: Token sequence (TokSequence or list of token IDs)

        Returns:
            symusic.Score object
        """
        score = self._tokenizer(tokens)
        score.tracks = sorted(score.tracks, key=lambda t: t.program)
        return score

    def save(self, save_path: str | Path) -> None:
        """
        Save the tokenizer configuration and vocabulary.

        Args:
            save_path: Directory path to save the tokenizer
        """
        self._tokenizer.save(Path(save_path))

    @classmethod
    def load(cls, load_path: str | Path) -> "RemiPlusTokenizer":
        """
        Load a previously saved tokenizer.

        Args:
            load_path: Directory path where tokenizer was saved

        Returns:
            RemiPlusTokenizer instance
        """
        instance = cls.__new__(cls)
        instance._tokenizer = REMI(params=Path(load_path))
        instance._config = instance._tokenizer.config
        return instance

    def __call__(self, input_data):
        """
        Allow the tokenizer to be called directly.

        Delegates to the underlying REMI tokenizer, which handles both
        tokenization (MIDI -> tokens) and decoding (tokens -> MIDI).

        Args:
            input_data: Either a MIDI file path or tokens (miditok.classes.TokSequence)

        Returns:
            Tokens if input was MIDI, or MIDI if input was tokens
        """
        if isinstance(input_data, (str, Path)):
            return self.tokenize(input_data)
        return self.decode(input_data)

    @property
    def vocab_size(self) -> int:
        """Get the full vocabulary size (including BPE tokens if trained)."""
        # len(tokenizer) returns the full vocab size including BPE
        return len(self._tokenizer)

    @property
    def base_vocab_size(self) -> int:
        """Get the base vocabulary size (before BPE)."""
        return len(self._tokenizer.vocab)


    def _load_and_fix_score(self, midi_path: str | Path) -> Score:
        """
        Load MIDI file and fix symusic's program parsing bug.

        - Removes empty tracks (no notes)
        - Fixes program numbers using mido as ground truth
        """

        score = Score(midi_path)
        mid = mido.MidiFile(midi_path)

        # Remove empty tracks
        score.tracks = [t for t in score.tracks if len(t.notes) > 0]

        # Get mido tracks with notes
        mido_tracks = []
        for track in mid.tracks:
            note_count = sum(1 for m in track if m.type == 'note_on' and m.velocity > 0)
            if note_count > 0:
                program = None
                for msg in track:
                    if msg.type == 'program_change':
                        program = msg.program
                        break
                mido_tracks.append({'notes': note_count, 'program': program})

        # Fix programs by matching note counts
        for mt, st in zip(mido_tracks, score.tracks):
            if mt['program'] is not None:
                st.program = mt['program']

        return score

    def tokenize(self, midi_path: str | Path):
        """
        Tokenize a single MIDI file.

        Args:
            midi_path: Path to the MIDI file

        Returns:
            TokSequence containing the tokens
        """
        score = self._load_and_fix_score(midi_path)
        return self._tokenizer(score)
