from typing import List, Dict
from .symbolic_model import TokenSet, TokenRange, REMIState, REMIConstraints, CounterpointSolver
from .tokenizer import RemiPlusTokenizer

from typing import List, Dict

def count_violations(token_sequence: List[int], tokenizer) -> Dict[str, int]:
    """
    Count violations of different rule types in a token sequence.
    Only counts violations of constraints that are actually enforced by the symbolic model.
    
    Args:
        token_sequence: List of token IDs
        tokenizer: RemiPlusTokenizer instance
        
    Returns:
        Dictionary with violation counts for each rule type
    """
    # Setup vocabulary mappings
    name_to_id = tokenizer.tokenizer.vocab
    id_to_name = {v: k for k, v in name_to_id.items()}
    
    # Instantiate components
    state = REMIState()
    bnf_constrainer = REMIConstraints(name_to_id, id_to_name)
    counterpoint_solver = CounterpointSolver(name_to_id, id_to_name)
    
    # Initialize violation counters
    violations = {
        'grammar': 0,                    # Wrong token type for grammar
        'position_exceeds_bar': 0,       # Position >= bar length
        'position_not_monotonic': 0,     # Position <= previous position
        'voice_unavailable': 0,          # Voice already playing
        'parallel_fifth': 0,             # Parallel fifth motion
        'parallel_octave': 0,            # Parallel octave motion
        # 'simultaneous_dissonance': 0,    # Dissonant simultaneity
        'invalid_token': 0,              # PAD or PitchDrum tokens
    }
    
    for token_id in token_sequence:
        token_name = id_to_name[token_id]
        token_type = token_name.split('_')[0]
        
        # Check for invalid tokens (PAD, PitchDrum)
        if token_type in ['PAD', 'PitchDrum']:
            violations['invalid_token'] += 1
            # Don't update state for invalid tokens
            continue
        
        # Check grammar violations
        allowed_types = bnf_constrainer.GRAMMAR.get(state.last_token_type, [])
        if token_type not in allowed_types:
            violations['grammar'] += 1
        
        # Check semantic violations based on token type
        if token_type == 'Position':
            position_val = int(token_name.split('_')[1])
            
            # Check if position exceeds bar length
            if state.bar_length_ticks is not None:
                if position_val >= state.bar_length_ticks:
                    violations['position_exceeds_bar'] += 1
            
            # Check monotonicity (must be strictly increasing)
            if state.current_position is not None:
                if position_val <= state.current_position:
                    violations['position_not_monotonic'] += 1
        
        elif token_type == 'Program':
            voice_id = int(token_name.split('_')[1])
            # Check if voice is available at current position
            if state.current_position is not None:
                if not state.can_play(voice_id):
                    violations['voice_unavailable'] += 1
        
        elif token_type == 'Pitch':
            # Check counterpoint violations using the solver
            if state.pending_program is not None:
                pitch_val = int(token_name.split('_')[1])
                
                # Get the full pitch domain for constraint checking
                pitch_domain = bnf_constrainer.domains.Pitch
                
                # Check each counterpoint rule
                if counterpoint_solver.is_parallel_fifth(state, pitch_domain, pitch_val):
                    violations['parallel_fifth'] += 1
                
                if counterpoint_solver.is_parallel_octave(state, pitch_domain, pitch_val):
                    violations['parallel_octave'] += 1
                
                # if counterpoint_solver.is_simultaneous_dissonance(state, pitch_domain, pitch_val):
                #    violations['simultaneous_dissonance'] += 1
        
        # Update state for next iteration
        state.update(token_id, token_name)
    
    return violations
