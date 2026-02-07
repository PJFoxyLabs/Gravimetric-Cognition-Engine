# The Gravimetric Cognition Engine

A PyTorch-based simulation engine for gravimetric cognition, encoding data into dynamic particle systems for advanced processing and analysis.

## Features

- Data encoding into particle-based representations
- Gravimetric simulation with mass, charge, and velocity dynamics
- PyTorch-powered computations for efficient processing
- Configurable particle capacity and simulation parameters
- Spectral Atomizer The Spectral Atomizer is a core component of the Gravimetric Cognition Engine, acting as a bridge between raw text data and the physics-based simulation. Unlike traditional tokenizers (which break text into words or subwords), it "atomizes" text into fundamental physical properties that particles can use in the simulation. Here's a breakdown of how it works:

Key Concept
Purpose: Converts text (as raw bytes) into "matter" — specifically, properties that define particles in the 3D simulation space. This allows the engine to treat language as physical entities with mass, charge, and frequency, enabling emergent behavior through physics laws.
No Tokenization: It skips linguistic parsing and directly maps byte values (0-255) to a continuous "spectrum" of frequencies. This creates a more primal, physics-driven representation.
How It Works
Frequency Mapping:

The class initializes a freq_map using torch.linspace(0.1, 2.0, 256, device=DEVICE), which creates 256 evenly spaced frequency values from 0.1 (low "bass" for control characters like null bytes) to 2.0 (high "treble" for high-value bytes).
Each byte value (0-255) is mapped to one of these frequencies, creating a spectral range for encoding.
Text-to-Matter Conversion (text_to_matter):

Input: A string (e.g., "hello").
Process:
Encodes the string to UTF-8 bytes (e.g., b'hello' → [104, 101, 108, 108, 111]).
Creates tensors for particle properties:
Mass: All particles get a base mass of 1.0 (representing equal "semantic importance" initially).
Charge: Alternates between +1.0 and -1.0 (e.g., +1, -1, +1, -1, +1). This creates "ionic-like" bonds, mimicking structural syntax in text (positive/negative charges attract/repel to maintain sequence integrity).
Frequency: Maps each byte to its corresponding frequency from freq_map (e.g., 'h' → ~1.2, 'e' → ~1.1).
Indices: Stores the original byte values for reference.
Output: A tuple of tensors (mass, charge, freq, indices) ready for particle injection into the simulation.
Matter-to-Char Conversion (matter_to_char):

Input: A single frequency value (from a particle's properties).
Process:
Finds the closest frequency in freq_map using torch.argmin(diff).
Converts the matching index back to a byte, then decodes it to a UTF-8 character.
Output: The nearest character (e.g., a frequency of 1.2 might decode back to 'h').
Purpose: Used during output generation, where the simulation "emits" characters based on accumulated frequencies at the Event Horizon.
Why "Spectral Atomizer"?
Spectral: Refers to the frequency spectrum, treating bytes as waves or tones rather than discrete tokens.
Atomizer: Breaks down text into atomic "particles" with physical attributes, ready for simulation.
This design allows the engine to process text as dynamic, interacting matter, where meaning emerges from physics (e.g., resonant frequencies grouping related concepts) rather than pre-trained patterns.
## Installation

1. Ensure Python 3.8+ is installed on your system.
2. Clone or download the project files.
3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the engine with:

```
python GravimetricCognitionEngine.py
```

The script will initialize the engine, encode sample data, and perform a simulation step, outputting results to the console.

## Dependencies

- PyTorch (>= 2.0)
- NumPy (>= 1.20)

## License

MIT License - see the copyright notice in `GravimetricCognitionEngine.py` for details.

## Troubleshooting

- If you encounter import errors, ensure all dependencies are installed in your Python environment.
- For GPU acceleration, modify the `get_device()` function to return the appropriate device (e.g., `torch.device("cuda")` if CUDA is available).
- The engine currently runs on CPU by default; DirectML support is available if installed for AMD GPUs.
