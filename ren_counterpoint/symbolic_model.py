from dataclasses import dataclass
from z3 import *
from typing import Optional, Tuple, Dict, List

@dataclass
class NoteEvent:
    pitch: int
    start_position: int  # absolute position in ticks from bar start, possibly negative
    end_position: int    # start + duration (may exceed bar length!)

class VoiceState:
    def __init__(self):
        self.current_note: Optional[NoteEvent] = None
        self.previous_note: Option[NoteEvent] = None

    def is_sounding_at(self, position: int) -> bool:
        """Check if this voice is playing at given position"""
        if self.current_note is None:
            return False
        return (self.current_note.start_position <= position <
                self.current_note.end_position)

    def can_start_note_at(self, position: int) -> bool:
        """Check if voice can start a new note at position"""
        if self.current_note is None:
            return True
        # New note can only start after current note ends
        return position >= self.current_note.end_position

    def start_note(self, pitch: int, position: int, duration: int):
        """Start a new note"""
        if self.current_note is not None and \
           self.current_note.end_position == position:
            self.previous_note = self.current_note
        else:
            self.previous_note = None

        self.current_note = NoteEvent(
                pitch=pitch,
                start_position=position,
                end_position=position + duration)

    def get_pitch_change_position(self) -> Optional[int]:
        """
        Get the position where the pitch last changed (i.e. where current_note started)
        Returns None if there's no current note or no previous note
        """
        if self.current_note is None or self.previous_note is None:
            return None
        return self.current_note.start_position

    def get_pitch_change(self) -> Optional[Tuple[int, int]]:
        if self.current_note is None or self.previous_note is None:
            return None
        return (self.previous_note, self.current_note)

    def advance_to_bar(self, new_bar_start_position: int):
        """Called when we cross a bar line"""
        if self.current_note is None:
            return

        # If note extends beyond current bar, adjust its coordinates
        # to be relative to the new bar
        if self.current_note.end_position > new_bar_start_position:
            # Note carries over - adjust to new bar's coordinate system
            self.current_note.start_position -= new_bar_start_position
            self.current_note.end_position -= new_bar_start_position
            if self.previous_note is not None:
                self.previous_note.start_position -= new_bar_start_position
                self.previous_note.end_position -= new_bar_start_position
        else:
            # Note finished in previous bar (or right at the bar line)
            # Keep it as previous_note, and adjust its coordinates
            self.previous_note = self.current_note
            self.previous_note.start_position -= new_bar_start_position
            self.previous_note.end_position -= new_bar_start_position
            self.current_note = None

class REMIState:
    def __init__(self, max_nb_voices=12):
        self.max_nb_voices = max_nb_voices

        # Voice states
        self.voices: List[VoiceState] = [VoiceState() for i in range(max_nb_voices)]

        # Bar/position tracking
        self.current_position: Optional[int] = None
        self.current_time_sig: Optional[str] = None
        # self.bar_length_ticks: Optional[int] = None

        # Grammar state (what we're expecting next)
        self.last_token_type: Optional[str] = None
        self.pending_program: Optional[int] = None  # Waiting for Pitch
        self.pending_pitch: Optional[int] = None    # Waiting for Duration

        # Flags for verification -----
        self.found_empty_bar = False
        self.found_non_zero_start = False

        # For empty bar detection
        self.notes_in_current_bar = 0

    @property
    def bar_length_ticks(self) -> Optional[int]:
        """Given current time sig calculate number of ticks until next bar"""
        if self.current_time_sig is None:
            return None

        # Time sig format: "3/4" means 3 quarter notes
        numerator, denominator = map(int, self.current_time_sig.split('/'))

        # # Convert to quarter notes (assuming denominator 4 = quarter, 8 = eighth)
        # if denominator == 4:
        #     beats = numerator
        # elif denominator == 8:
        #     beats = numerator / 2  # 3/8 = 1.5 quarter notes
        # else:
        #     raise ValueError(f"Unexpected time signature: {self.current_time_sig}")

        # Max position is (beats * 24) - 1
        # e.g., 4/4 = 4 beats = 96 ticks, so max position is 95
        # Position 96 would be the downbeat of next bar (Position_0)
        return int(numerator * 24)


    def parse_duration(self, duration_str: str) -> int:
        """Parse 'Duration_X.Y.Z' into ticks"""
        # Remove 'Duration_' prefix
        parts = duration_str.replace('Duration_', '').split('.')
        beats, ticks, resolution = map(int, parts)

        # Total ticks = beats * resolution + ticks
        return beats * resolution + ticks

    def update(self, token_id: int, token_name: str):
        """Update state with new token"""
        token_type, token_val = token_name.split('_')

        if token_type == 'BOS':
            self.last_token_type = 'BOS'

        elif token_type == 'PAD':
            raise ValueError(f"PAD token found in a sequence.  Should be impossible.")

        elif token_type == 'Bar':
            # For empty bar flag -----
            if self.last_token_type == 'TimeSig' and self.notes_in_current_bar == 0:
                self.found_empty_bar = True

            # Crossing a bar line - advance all voices
            if self.bar_length_ticks is not None:
                for voice in self.voices:
                    voice.advance_to_bar(self.bar_length_ticks)

            # Reset bar-level state
            self.current_position = None
            self.notes_in_current_bar = 0 # -------
            self.last_token_type = 'Bar'

        elif token_type == 'TimeSig':
            self.current_time_sig = token_val
            self.last_token_type = 'TimeSig'
            # self.current_position has already been set to None by Bar logic

        elif token_type == 'Position':
            new_position = int(token_val)

            # Check if first position of bar is non-zero
            if self.last_token_type == 'TimeSig' and new_position != 0:
                self.found_non_zero_start = True

            self.current_position = new_position
            self.last_token_type = 'Position'

        elif token_type == 'Program':
            self.pending_program = int(token_val)
            self.last_token_type = 'Program'

        elif token_type == 'Pitch':
            self.pending_pitch = int(token_val)
            self.last_token_type = 'Pitch'

        elif token_type == 'Duration':
            # We now have a complete note: (position, program, pitch, duration)
            duration_ticks = self.parse_duration(token_name)

            # Record the note in the voice
            self.voices[self.pending_program].start_note(
                pitch=self.pending_pitch,
                position=self.current_position,
                duration=duration_ticks)

            # For setting empty bar flag -----
            self.notes_in_current_bar += 1

            # Clear pending
            self.pending_program = None
            self.pending_pitch = None
            self.last_token_type = 'Duration'

        elif token_type == 'EOS':
            if self.last_token_type == 'TimeSig' and self.notes_in_current_bar == 0:
                self.found_empty_bar = True
            self.last_token_type = 'EOS'

    def can_play(self, voice_id: int) -> bool:
        """
        Check if a voice can start a note at current position.
        N.B. voice_id is in [0, max_nb_voices[, not the associated token"""
        if self.current_position is None:
            return False
        return self.voices[voice_id].can_start_note_at(self.current_position)



@dataclass
class TokenRange:
    """Represents a contiguous range of token IDs [start, end["""
    start: int
    end: int
    name: str

    def __contains__(self, token_id: int) -> bool:
        return self.start <= token_id < self.end

    def to_tuple(self) -> Tuple[int, int]:
        """Return (start, end) for Z3 integration"""
        return (self.start, self.end)

    def size(self) -> int:
        return self.end - self.start

    def has_member(self, x: z3.z3.ArithRef) -> z3.z3.BoolRef:
        """Enforce that x is in the token range"""
        return self.start <= x < self.end

@dataclass
class TokenSet:
    """Represents a possibly noncontiguous range of token IDs"""
    ids: set
    name: str

    def __contains__(self, token_id: int) -> bool:
        return token_id in self.ids

    def size(self) -> int:
        return len(self.ids)

    def has_member(self, x: z3.z3.ArithRef) -> z3.z3.BoolRef:
        """Enforce that x is a member of the set of ids"""
        return Or(*(x == id for id in self.ids))

@dataclass
class VocabDomains:
    """Pre-computed token ID domains for each token type"""
    BOS: TokenSet
    EOS: TokenSet
    PAD: TokenSet
    Bar: TokenSet
    TimeSig: TokenSet            # Only time signatures the model has seen
    Position: TokenRange         # Positions 0-143 at most, to be filtered by TimeSig later
    Program: TokenRange          # Programs 0-11
    Pitch: TokenRange            # Pitches 21-109
    Duration: TokenRange         # All durations
    PitchDrum: TokenRange        # ALWAYS INVALID - marked for exclusion

    @classmethod
    def from_vocab(cls, name_to_id: Dict[str, int], id_to_name: Dict[int, str]):
        """
        Build token singleton ids, sets, and ranges from vocab
        """
        def find_range(prefix: str) -> TokenRange:
            """Find contiguous range for a token type"""
            matching = [
                tid for tid, name in id_to_name.items()
                if name.startswith(prefix)]
            return TokenRange(min(matching), max(matching)+1, prefix.strip('_'))

        # Valid time signatures only
        valid_timesigs = {'3/8', '6/4', '3/4', '2/4', '4/4'}
        timesig_set = TokenSet(
            {name_to_id['TimeSig_'+sig] for sig in valid_timesigs}, 'TimeSig'
        )
        # Positions 0-143 only (6/4 is max: 6 beats * 24 ticks)
        position_ids = [
            tid for tid, name in id_to_name.items()
            if name.startswith('Position_') and 0 <= int(name.split('_')[1]) < 144
        ]
        position_range = TokenRange(
            min(position_ids), max(position_ids)+1, 'Position'
        )

        return cls(
            BOS=TokenSet({name_to_id['BOS_None']}, 'BOS_None'),
            EOS=TokenSet({name_to_id['EOS_None']}, 'EOS_None'),
            PAD=TokenSet({name_to_id['PAD_None']}, 'PAD_None'),
            Bar=TokenSet({name_to_id['Bar_None']}, 'Bar_None'),
            TimeSig=timesig_set,
            Position=position_range,
            Program=find_range('Program_'),
            Pitch=find_range('Pitch_'),
            Duration=find_range('Duration_'),
            PitchDrum=find_range('PitchDrum_'),
        )



class REMIConstraints:
    """
    Constraint checker for syntax and semantics of REMI+ generation
    Returns valid token domains that can be used for masking or Z3 solving
    """
    GRAMMAR = {
        None: ['BOS'],
        'BOS': ['Bar'],
        'Bar': ['TimeSig'],
        'TimeSig': ['Position', 'Bar', 'EOS'],  # Position | empty bar | end
        'Position': ['Program'],
        'Program': ['Pitch'],
        'Pitch': ['Duration'],
        'Duration': ['Program', 'Position', 'Bar', 'EOS'],
    }

    def __init__(self, name_to_id: Dict[str, int], id_to_name: Dict[int, str]):
        self.vocab_size = len(id_to_name)
        self.id_to_name = id_to_name
        self.name_to_id = name_to_id
        self.domains = VocabDomains.from_vocab(self.name_to_id, self.id_to_name)

    def get_valid_domains(self, state) -> List[TokenRange | TokenSet]:
        """
        Get valid token ranges based on grammar and basic semantic constraints
        Returns list of TokenRange objects that can be:
        1. Directly used to create masks
        2. Passed to Z3 solver for further constraint solving
        """
        # Get grammar-allowed types
        allowed_types = self.GRAMMAR.get(state.last_token_type, [])

        valid = []
        for token_type in allowed_types:
            domain = getattr(self.domains, token_type)

            # Apply semantic filtering to potentially narrow the domains
            filtered_domains = self._filter_semantic(token_type, domain, state)
            valid.extend(filtered_domains)

        return valid

    def _filter_semantic(
        self,
        token_type: str,
        domain: TokenRange | TokenSet,
        state: REMIState
    ) -> List[TokenRange | TokenSet]:
        """
        Invoke a method to constrain token domains, if applicable
        Returns list of domains (to support TokenRange split into multiple subranges)
        """
        if token_type == 'Position':
            return self._filter_position_range(domain, state)
        elif token_type == 'Program':
            return self._filter_program_range(domain, state)
        else:
            # No filtering needed - return full range
            return [domain]

    def _filter_position_range(
        self,
        domain: TokenRange,
        state: REMIState
    ) -> List[TokenRange | TokenSet]:
        """
        Filter position range by:
        1. Bar length (max position depends on time signature)
        2. Monotonicity (must be > current_position)
        """
        assert state.current_time_sig is not None, f"Position without time sig: {state}"

        if state.last_token_type == "TimeSig":
            # After TimeSig, before first position - any position OK
            return [domain]

        max_pos = state.bar_length_ticks - 1
        current = state.current_position if state.current_position is not None else -1 # example: right after TimeSig

        # New position must be: current < pos <= max_pos
        min_valid_pos = current + 1
        max_valid_pos = max_pos

        if min_valid_pos > max_valid_pos:
            # No valid positions (bar is full)
            return []

        # Convert position values to token IDs
        # Position_N maps to token ID = position_range.start + N
        min_token_id = domain.start + min_valid_pos
        max_token_id = domain.start + max_valid_pos

        # # Clamp to actual range bounds - should NOT be necessary
        # min_token_id = max(min_token_id, position_range.start)
        # max_token_id = min(max_token_id, position_range.end)
        assert min_token_id >= domain.start, \
            f"min_token_id, {min_token_id}, too small for range {domain}"
        assert max_token_id <= domain.end, \
            f"max_token_id, {max_token_id}, too large for range {domain}"

        # if min_token_id > max_token_id:
        #     return []
        assert min_token_id <= max_token_id, \
                f'''Given (min_valid_pos, max_valid_pos)=({min_valid_pos},{max_valid_pos}),
                    (min_token_id, max_token_id)=({min_token_id},{max_token_id})
                    should not be empty'''

        return [TokenRange(min_token_id, max_token_id + 1, 'Position')]

    def _filter_program_range(
        self,
        domain: TokenRange,
        state
    ) -> List[TokenSet]:
        """
        Filter program range by voice availability
        """
        assert state.current_position is not None, f"_filter_program_range requires a Position to have already been set"
        available_voices = {domain.start + voice_id for voice_id in range(state.max_nb_voices) if state.can_play(voice_id)}
        return [TokenSet(available_voices, 'Program')]

  # def get_invalid_mask(self, state, z3_invalid_ids: Optional[List[int]] = None) -> np.ndarray:
    #   """
    #   Create boolean mask: True for INVALID tokens

    #   Args:
    #       state: Current generation state

    #   Returns:
    #       Boolean array of shape [vocab_size] where True = invalid
    #   """
    #   # Start with all tokens invalid
    #   mask = np.ones(self.vocab_size, dtype=bool)

    #   # Mark valid domains as False
    #   valid = self.get_valid_domains(state)
    #   for domain in valid:
    #     if isinstance(domain, TokenRange):
    #       mask[ domain.start : domain.end ] = False
    #     elif isinstance(domain, TokenSet):
    #       mask[ list(domain.ids) ] = False
    #     else:
    #       raise ValueError(f"Unknown domain instance: {domain}")

    #   # PitchDrum is already excluded - nothing to do

    #   return mask

    # def mask_logits(
    #     self,
    #     logits: np.ndarray,
    #     state,
    # ) -> np.ndarray:
    #     """
    #     Apply constraints: set invalid token logits to -inf

    #     Args:
    #         logits: Model logits of shape [vocab_size]
    #         state: Current generation state

    #     Returns:
    #         Masked logits with invalid tokens set to -inf
    #     """
    #     masked = logits.copy()
    #     invalid_mask = self.get_invalid_mask(state)
    #     masked[invalid_mask] = float('-inf')
    #     return masked

    def get_valid_token_ids(self, state) -> List[int]:
        """
        Get list of all valid token IDs (for debugging/analysis)
        Less efficient than range-based masking, but useful for inspection
        """
        valid = self.get_valid_domains(state)
        valid_ids = []
        for domain in valid:
            if isinstance(domain, TokenRange):
                valid_ids.extend(range(domain.start, domain.end))
            elif isinstance(domain, TokenSet):
                valid_ids.extend(domain.ids)
            else:
                raise ValueError(f"Unknown domain instance: {domain}")

        # There should NOT be any PitchDrum tokens!
        assert(not any(self.id_to_name[id].startswith("PitchDrum") for id in valid_ids))

        return valid_ids

class CounterpointSolver:
    """
    Uses Z3 to solve for forbidden pitches
    """
    def __init__(self, name_to_id: Dict[str, int], id_to_name: Dict[int, str]):
        self.name_to_id = name_to_id
        self.id_to_name = id_to_name

        

    def parallel_fifth(
        self,
        state: REMIState,
        pitch_domain: TokenRange,
        pitch: z3.z3.ArithRef) -> z3.z3.BoolRef:
        """Create constraints to solve for parallel fifths"""

        current_voice = state.pending_program
        assert current_voice is not None, f"solver invoked with no pending program"

        this_curr = state.voices[current_voice].current_note
        if this_curr is None:
            return False

        constraints = []
        for other_voice_idx in range(state.max_nb_voices):
            if other_voice_idx == current_voice:
                continue

            other_voice = state.voices[other_voice_idx]
            if other_voice.get_pitch_change_position() != state.current_position:
                continue

            other_change = other_voice.get_pitch_change()
            if other_change is None:
                continue

            other_prev, other_curr = other_change
            if other_prev == other_curr:
                continue

            if (this_curr.pitch - other_prev.pitch) % 12 == 7:
                constraints.append((pitch - other_curr.pitch) % 12 == 7)
            elif (other_prev.pitch - this_curr.pitch) % 12 == 7:
                constraints.append((other_curr.pitch - pitch) % 12 == 7)

        return Or(constraints) if constraints else False

    def parallel_octave(
        self,
        state: REMIState,
        pitch_domain: TokenRange,
        pitch: z3.z3.ArithRef) -> z3.z3.BoolRef:
        """Create constraints to solve for parallel octaves"""

        current_voice = state.pending_program
        assert current_voice is not None, f"solver invoked with no pending program"

        this_curr = state.voices[current_voice].current_note
        if this_curr is None:
            return False

        constraints = []
        for other_voice_idx in range(state.max_nb_voices):
            if other_voice_idx == current_voice:
                continue

            other_voice = state.voices[other_voice_idx]
            if other_voice.get_pitch_change_position() != state.current_position:
                continue

            other_change = other_voice.get_pitch_change()
            if other_change is None:
                continue

            other_prev, other_curr = other_change
            if other_prev == other_curr:
                continue

            if (this_curr.pitch - other_prev.pitch) % 12 != 0:
                continue

            constraints.append((pitch - other_curr.pitch) % 12 == 0)

        return Or(constraints) if constraints else False

    def simultaneous_dissonance(
        self,
        state: REMIState,
        pitch_domain: TokenRange,
        pitch: z3.z3.ArithRef) -> z3.z3.BoolRef:
    
        current_voice = state.pending_program
        assert current_voice is not None, f"solver invoked with no pending program"
    
        constraints = []
        sounding = [
            state.voices[idx].current_note.pitch for idx in range(state.max_nb_voices)
            if idx != current_voice and \
               state.voices[idx].is_sounding_at(state.current_position)]
    
        # lowest_pitch = min(sounding) if sounding else None
        # if lowest_pitch is not None:
        #   constraints.append(
        #     And(pitch > lowest_pitch, (pitch - lowest_pitch) % 12 == 5)
        #   )
    
    
        for other_voice_idx in range(state.max_nb_voices):
            if other_voice_idx == current_voice:
                continue
    
            other_voice = state.voices[other_voice_idx]
            if other_voice.get_pitch_change_position() != state.current_position:
                continue
    
            other_change = other_voice.get_pitch_change()
            if other_change is None:
                continue
    
            _, other_curr = other_change
            interval = Abs(pitch - other_curr.pitch) % 12
            for dissonance in (1,2,6,10,11):
                constraints.append(interval == dissonance)
    
            #if lowest_pitch is not None:
            #  constraints.append(And(pitch <= lowest_pitch, (other_curr.pitch - pitch) % 12 == 5))
            #  # constraints.append(And(other_curr.pitch <= lowest_pitch, (pitch - other_curr.pitch) % 12 == 5))
    
        return Or(constraints) if constraints else False
    
    
    def solve_pitch_constraints(
        self,
        state: REMIState,
        pitch_token_domain: TokenRange
        ) -> TokenSet:
        """Returns a set of pitch tokens that violate pitch constraints"""
        pitch = Int('pitch')
        pitch_domain = TokenRange(
            int(self.id_to_name[pitch_token_domain.start].replace('Pitch_','')),
            int(self.id_to_name[pitch_token_domain.end-1].replace('Pitch_','')),
            'Pitch')
    
        solver = Solver()
        solver.add(And(
            pitch_domain.start <= pitch, pitch < pitch_domain.end))
        solver.add(Or(
            self.parallel_fifth(state, pitch_domain, pitch),
            self.parallel_octave(state, pitch_domain, pitch),
            self.simultaneous_dissonance(state, pitch_domain, pitch)
        ))
    
        forbidden_pitches = []
        while solver.check() == sat:
            model = solver.model()
            value = model[pitch].as_long()
            forbidden_pitches.append(value)
            solver.add(pitch != value)
    
        return TokenSet({self.name_to_id[f'Pitch_{x}'] for x in forbidden_pitches} , 'Pitch')

    def is_parallel_fifth(self, state: REMIState, pitch_domain: TokenRange, pitch_value: int) -> bool:
        """Check if a specific pitch value creates a parallel fifth"""

        pitch = Int('pitch')  
        solver = Solver()
        solver.add(pitch == pitch_value)
        solver.add(self.parallel_fifth(state, pitch_domain, pitch)) 
        return solver.check() == sat
    
    def is_parallel_octave(self, state: REMIState, pitch_domain: TokenRange, pitch_value: int) -> bool:
        """Check if a specific pitch value creates a parallel octave"""
        pitch = Int('pitch')
        solver = Solver()
        solver.add(pitch == pitch_value)
        solver.add(self.parallel_octave(state, pitch_domain, pitch))
        return solver.check() == sat
        
    def is_simultaneous_dissonance(self, state: REMIState, pitch_domain: TokenRange, pitch_value: int) -> bool:
        """Check if a specific pitch value creates a dissonance"""
        pitch = Int('pitch')
        solver = Solver()
        solver.add(pitch == pitch_value)
        solver.add(self.simultaneous_dissonance(state, pitch_domain, pitch))
        return solver.check() == sat
