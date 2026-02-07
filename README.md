# The Gravimetric Cognition Engine

A PyTorch-based simulation engine for gravimetric cognition, encoding data into dynamic particle systems for advanced processing and analysis.

## Features

- Data encoding into particle-based representations
- Gravimetric simulation with mass, charge, and velocity dynamics
- PyTorch-powered computations for efficient processing
- Configurable particle capacity and simulation parameters

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
