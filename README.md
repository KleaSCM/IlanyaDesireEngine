# Ilanya Desire Engine

An advanced mathematical system for managing desire space dynamics using PyTorch, attention mechanisms, and sophisticated mathematical models.

## Overview

The Desire Engine is a sophisticated system that creates, manages, and evolves desires based on trait activations from the existing Ilanya neural network trait system. It uses advanced mathematical models including:

- **Neural Embeddings**: High-dimensional representations of desires
- **Attention Mechanisms**: Transformer-style attention for desire interactions
- **Reinforcement Learning**: Dynamic strength adjustment based on trait reinforcement
- **Time-based Decay**: Natural weakening of desires over time
- **Goal Assessment**: Evaluation of desire potential to become goals

## Key Features

### 🧠 Advanced Mathematical Models
- **Desire Embeddings**: 64-dimensional neural representations of desires
- **Attention Mechanisms**: 8-head transformer attention for desire interactions
- **Sigmoid Reinforcement**: Sophisticated reinforcement using sigmoid functions
- **Exponential Decay**: Natural time-based decay of desire strength

### 🔄 Dynamic Desire Management
- **Creation**: Desires emerge from positive trait activations
- **Reinforcement**: Strengthening through repeated positive trait changes
- **Pruning**: Automatic removal of weak, unreinforced desires
- **Goal Assessment**: Identification of desires ready to become goals

### 🎯 Integration with Trait System
- **Trait Input**: Processes trait states from the existing neural network
- **Positive Filtering**: Only positive trait changes create desires
- **Identity Protection**: Respects protected traits (sexual orientation, gender identity, etc.)
- **Big Five Integration**: Works with Big Five personality traits

## Architecture

### Core Components

1. **Desire Class**: Mathematical representation of individual desires
2. **DesireEmbedding**: Neural network for embedding desires into mathematical space
3. **DesireSpaceAttention**: Transformer attention for desire interactions
4. **DesireEngine**: Main orchestrator for desire space management

### Mathematical Foundations

#### Desire Strength Calculation
```
strength = base_strength * (1 + reinforcement_bonus)
reinforcement_bonus = reinforcement_strength * sigmoid(reinforcement_count - 3)
```

#### Time-based Decay
```
decay_factor = exp(-decay_rate * hours_passed)
new_strength = current_strength * decay_factor
```

#### Goal Potential Assessment
```
goal_potential = min(1.0, reinforcement_count / 10.0)
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd IlanyaDesireEngine
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure the IlanyaNN directory is available** (contains the trait system)

## Usage

### Basic Usage

```python
from desire_engine import DesireEngine, DesireEngineConfig
from IlanyaNN.trait_types import TraitType
from IlanyaNN.trait_state import TraitState

# Initialize the desire engine
config = DesireEngineConfig(
    reinforcement_threshold=0.3,
    max_desires=50
)
desire_engine = DesireEngine(config)

# Process trait activations
trait_states = {
    TraitType.OPENNESS: TraitState(
        trait_type=TraitType.OPENNESS,
        current_value=0.8,
        previous_value=0.6,
        change_rate=0.2
    )
}

results = desire_engine.process_trait_activations(trait_states)
print(f"New desires: {results['new_desires']}")
print(f"Goal candidates: {results['goal_candidates']}")
```

### Advanced Usage

```python
# Get neural embeddings
embeddings = desire_engine.get_desire_embeddings()

# Compute attention weights
updated_embeddings, attention_weights = desire_engine.compute_desire_attention()

# Get comprehensive summary
summary = desire_engine.get_desire_summary()

# Save/load state
desire_engine.save_state("desire_state.json")
desire_engine.load_state("desire_state.json")
```

## Running the Demo

```bash
python desire_engine_demo.py
```

The demo shows:
1. **Desire Creation**: From positive trait activations
2. **Reinforcement**: Through repeated trait changes
3. **Decay & Pruning**: Time-based weakening and removal
4. **Negative Traits**: How negative changes don't create desires
5. **Neural Embeddings**: Attention mechanisms in action
6. **State Persistence**: Saving and loading desire states

## Configuration

### DesireEngineConfig Parameters

- **desire_dim**: Embedding dimension (default: 64)
- **hidden_dim**: Hidden layer dimension (default: 128)
- **num_attention_heads**: Attention heads (default: 8)
- **reinforcement_threshold**: Minimum change for reinforcement (default: 0.3)
- **max_desires**: Maximum number of active desires (default: 50)
- **pruning_threshold**: Strength threshold for pruning (default: 0.05)
- **goal_candidate_threshold**: Threshold for goal candidacy (default: 0.8)

## Desire States

1. **ACTIVE**: Currently active desire
2. **REINFORCED**: Strongly reinforced desire
3. **GOAL_CANDIDATE**: Ready to become a goal
4. **WEAKENING**: Losing strength, may be pruned
5. **PRUNED**: Has been removed from active space

## Integration with Trait System

The Desire Engine integrates seamlessly with the existing Ilanya trait system:

- **Input**: TraitState objects from the neural network
- **Processing**: Identifies positive trait changes
- **Output**: Desire creation, reinforcement, and goal assessment
- **Protection**: Respects identity-protected traits

## Mathematical Models

### Reinforcement Function
Uses sigmoid function for smooth reinforcement:
```
f(x) = 1 / (1 + e^(-x + 3))
```

### Attention Mechanism
Transformer-style multi-head attention:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

### Embedding Fusion
Multi-layer neural network for combining desire features:
```
Fusion(x) = LayerNorm(Linear(ReLU(Linear(x))))
```

## Future Enhancements

- **Emotional Field Integration**: Weighting desires by emotional state
- **Goal Engine Integration**: Seamless transition to goal processing
- **Advanced Attention**: Hierarchical attention mechanisms
- **Temporal Modeling**: LSTM/GRU for temporal desire evolution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Author

KleaSCM - KleaSCM@gmail.com

## Version

0.1.0 - Initial release with core functionality 