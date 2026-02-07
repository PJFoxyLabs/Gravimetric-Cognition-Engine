# The Gravimetric Cognition Engine (v1.0 - Genesis)
# Copyright (c) 2026 Phillip Fox (PJFoxy)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ARCHITECTURE: Resonant N-Body Field Computer
# DEPENDENCIES: PyTorch (Compute Only). NO Transformers. NO Pre-training.

import torch
import math
import time
import sys
import numpy as np
from collections import deque

# --- Hardware Interface ---
def get_device():
    # Force CPU fallback for compatibility
    return torch.device("cpu")

DEVICE = get_device()
print(f"🌌 WELLS ENGINE ONLINE. Substrate: {DEVICE}")

# --- Universal Constants ---
G_CONST = 20.0          # Gravitational Constant (Semantic Pull)
K_ELEC = 50.0           # Coulomb Constant (Structural Repulsion)
C_LIGHT = 10.0          # Speed of Causality
DT = 0.05               # Planck Time
DAMPING = 0.985         # Vacuum Viscosity (Prevents explosion)
RESONANCE_K = 15.0      # Harmonic Attraction Strength
HEBBIAN_RATE = 0.002    # Rate of Memory Solidification
BOX_SIZE = 100.0        # Universe Boundary
EVENT_HORIZON = 5.0     # Output Radius

# --- 1. The Spectral Atomizer (No Tokenizer) ---
# Converts Raw Bytes <-> Physics Properties
class SpectralAtomizer:
    def __init__(self):
        # We map the 256 byte values to the 0-1 frequency spectrum
        # 0 = Low Bass (Control chars), 255 = High Treble
        self.freq_map = torch.linspace(0.1, 2.0, 256, device=DEVICE)
        
    def text_to_matter(self, text: str):
        """Converts string to particle properties."""
        bytes_data = text.encode('utf-8')
        indices = torch.tensor([b for b in bytes_data], device=DEVICE, dtype=torch.long)
        
        N = len(indices)
        if N == 0: return None
        
        # Properties
        mass = torch.ones(N, device=DEVICE) * 1.0
        # Charge alternates to create structural bonds (ionic-like text structure)
        charge = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(N)], device=DEVICE)
        freq = self.freq_map[indices]
        
        return mass, charge, freq, indices

    def matter_to_char(self, frequency):
        """Decodes a frequency back to the nearest byte."""
        diff = torch.abs(self.freq_map - frequency)
        idx = torch.argmin(diff).item()
        try:
            return bytes([idx]).decode('utf-8')
        except:
            return "?" # Invalid byte sequence

# --- 2. The Wells Field (The Brain) ---
class WellsField:
    def __init__(self, capacity=16384):
        self.capacity = capacity
        
        # --- State Tensors (The Universe) ---
        # [N, 3] Spatial Position (x, y, z)
        self.pos = torch.zeros((capacity, 3), device=DEVICE)
        # [N, 3] Velocity
        self.vel = torch.zeros((capacity, 3), device=DEVICE)
        
        # --- Properties ---
        self.mass = torch.zeros(capacity, device=DEVICE)
        self.charge = torch.zeros(capacity, device=DEVICE)
        self.freq = torch.zeros(capacity, device=DEVICE) # The "Meaning"
        self.active = torch.zeros(capacity, dtype=torch.bool, device=DEVICE)
        
        # --- Hebbian Metric (Long Term Memory) ---
        # A sparse adjacency matrix defining "Hardened Space"
        # Since full N^2 is expensive, we approximate with dynamic bonds
        # Here we simplify: we store a "Home" frequency for the region
        
        self.active_count = 0
        self.atomizer = SpectralAtomizer()
        
        # The Emitter (The Mouth)
        self.emitter_accum = 0.0
        self.last_emission_time = 0

    def inject(self, text):
        """Spawns particles from text input."""
        mass, charge, freq, _ = self.atomizer.text_to_matter(text)
        if mass is None: return
        
        N = len(mass)
        # Find free slots
        free_indices = torch.where(~self.active)[0]
        if len(free_indices) < N:
            # Universe full: Entropy death of oldest particles (simplified rollover)
            # In a real run, we'd export to disk
            reset_n = N - len(free_indices) + 100
            active_idx = torch.where(self.active)[0]
            # Kill random particles to make space (Forgetting)
            kill_idx = active_idx[torch.randperm(len(active_idx))[:reset_n]]
            self.active[kill_idx] = False
            self.mass[kill_idx] = 0
            free_indices = torch.where(~self.active)[0]
            
        indices = free_indices[:N]
        
        # Initialize State
        # Inject at the "White Hole" (Top of the box)
        spawn_spread = 5.0
        self.pos[indices] = torch.randn(N, 3, device=DEVICE) * spawn_spread
        self.pos[indices, 1] += 40.0 # High Y
        
        # Initial Velocity: Downward stream
        self.vel[indices] = torch.randn(N, 3, device=DEVICE) * 0.1
        self.vel[indices, 1] -= 2.0 
        
        # Properties
        self.mass[indices] = mass
        self.charge[indices] = charge
        self.freq[indices] = freq
        self.active[indices] = True
        
        self.active_count += N
        print(f"🌌 Injected '{text}' ({N} particles). Active: {self.active_count}")

    def step(self):
        """
        The Fundamental Force Loop.
        No Neural Net. Just N-Body Physics with Semantic Resonance.
        """
        if self.active_count == 0: return

        active_idx = torch.where(self.active)[0]
        N = len(active_idx)
        
        # Pointers to active data
        p = self.pos[active_idx]
        v = self.vel[active_idx]
        m = self.mass[active_idx]
        q = self.charge[active_idx]
        f = self.freq[active_idx]
        
        # --- 1. Compute Distance Matrix (Geometry) ---
        # r_ij = p_i - p_j
        # We need N^2 interactions. 
        # Optimization: We assume N < 5000 for real-time without Barnes-Hut in this demo.
        
        # Expand for broadcast: (N, 1, 3) - (1, N, 3) = (N, N, 3)
        diff = p.unsqueeze(1) - p.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2) + 0.1 # Softening epsilon
        dist = torch.sqrt(dist_sq)
        
        # Directions: (N, N, 3)
        dirs = diff / dist.unsqueeze(2)
        
        # --- 2. Compute Forces ---
        
        # A. Gravity (Mass Attraction)
        # F = G * m1 * m2 / r^2
        f_grav_mag = (G_CONST * m.unsqueeze(1) * m.unsqueeze(0)) / dist_sq
        
        # B. Electrostatics (Structure)
        # F = K * q1 * q2 / r^2 (Likes repel, opposites attract)
        f_elec_mag = -(K_ELEC * q.unsqueeze(1) * q.unsqueeze(0)) / dist_sq
        
        # C. Resonant Force (The "Attention" Replacement)
        # Particles with similar frequencies ATTRACT strongly.
        # This groups letters into words, and words into concepts.
        # F_res = K_res * Cos(f1 - f2) / r
        # Similar freq (diff ~ 0) -> cos=1 -> Attract
        freq_diff = f.unsqueeze(1) - f.unsqueeze(0)
        # We map frequency difference to a phase. Close freq = high attraction.
        resonance = torch.cos(freq_diff * 10.0) # Sharp peaks
        f_res_mag = (RESONANCE_K * resonance) / dist
        
        # Total Force Magnitude
        # Zero out diagonal (self-interaction)
        eye_mask = 1.0 - torch.eye(N, device=DEVICE)
        f_total_mag = (f_grav_mag + f_elec_mag + f_res_mag) * eye_mask
        
        # Sum Forces: (N, N, 1) * (N, N, 3) -> (N, N, 3) -> Sum dim 1 -> (N, 3)
        acc = torch.sum(f_total_mag.unsqueeze(2) * dirs, dim=1)
        
        # Add Central Attractor (The "Core Meaning" / Black Hole at 0,0,0)
        # Pulls everything to the center so they interact
        center_dist = torch.norm(p, dim=1, keepdim=True) + 1.0
        acc -= (p / center_dist) * 5.0 # Constant centripetal pull
        
        # --- 3. Integration (Symplectic) ---
        v_new = (v + acc * DT) * DAMPING
        p_new = p + v_new * DT
        
        # --- 4. Boundary Conditions ---
        # Reflective Box
        box_mask = torch.abs(p_new) > BOX_SIZE
        v_new[box_mask] *= -0.5 # Inelastic collision
        p_new[box_mask] = torch.clamp(p_new[box_mask], -BOX_SIZE, BOX_SIZE)
        
        # Update State
        self.vel[active_idx] = v_new
        self.pos[active_idx] = p_new
        
        # --- 5. Hebbian Crystallization (Learning) ---
        # If particles are very close and resonant, increase their mass (importance)
        # This makes frequent associations "heavier" and harder to break.
        close_mask = (dist < 1.0) & (resonance > 0.9) & (eye_mask.bool())
        if close_mask.any():
            # Get indices of interacting pairs
            # Boost mass slightly
            m_boost = torch.sum(close_mask.float(), dim=1) * HEBBIAN_RATE
            self.mass[active_idx] += m_boost
            # Cap mass
            self.mass[active_idx] = torch.clamp(self.mass[active_idx], 1.0, 50.0)

    def spectrometer_readout(self):
        """
        The Output Mechanism.
        Checks for particles falling into the Event Horizon (Center).
        If enough mass of a specific frequency accumulates, it 'emits' that char.
        """
        if self.active_count == 0: return None
        
        active_idx = torch.where(self.active)[0]
        p = self.pos[active_idx]
        
        # Check distance to origin
        r = torch.norm(p, dim=1)
        
        # Event Horizon: Particles very close to 0,0,0
        horizon_mask = r < EVENT_HORIZON
        
        if not horizon_mask.any(): return None
        
        in_horizon_idx = active_idx[horizon_mask]
        
        # Calculate the dominant frequency in the horizon
        # Weighted by Mass (Importance) and Velocity (Energy)
        horizon_mass = self.mass[in_horizon_idx]
        horizon_freq = self.freq[in_horizon_idx]
        
        # Weighted average frequency
        dominant_freq = torch.sum(horizon_freq * horizon_mass) / torch.sum(horizon_mass)
        
        # Accumulate "Emission Energy"
        energy_in = torch.sum(horizon_mass).item()
        self.emitter_accum += energy_in
        
        # Threshold to emit a character (Quantum of Meaning)
        EMISSION_THRESHOLD = 50.0
        
        if self.emitter_accum > EMISSION_THRESHOLD:
            self.emitter_accum = 0.0 # Discharge
            
            # Decode
            char = self.atomizer.matter_to_char(dominant_freq)
            
            # Kick particles out of horizon (Evaporation)
            # This prevents repeating the same char forever (loop penalty)
            kick_vec = torch.randn(len(in_horizon_idx), 3, device=DEVICE) * 20.0
            self.vel[in_horizon_idx] += kick_vec
            self.pos[in_horizon_idx] += kick_vec * 0.1
            
            return char
            
        return None

# --- 3. The Interface ---
def run_wells():
    engine = WellsField()
    
    print("\n--- WELLS GRAVIMETRIC INTERFACE ---")
    print("Direct matter injection ready.")
    print("Physics running in real-time background loop.")
    print("Type 'exit' to collapse universe.")
    
    # Pre-seed with some primordial soup (random particles) to allow transmission
    engine.inject("SYSTEM_ONLINE_")
    
    import threading
    
    # Run physics in background thread
    running = True
    
    def physics_loop():
        while running:
            engine.step()
            output = engine.spectrometer_readout()
            if output:
                # Direct brain-to-terminal output
                sys.stdout.write(output)
                sys.stdout.flush()
            time.sleep(0.001) # Max speed
            
    t = threading.Thread(target=physics_loop)
    t.daemon = True
    t.start()
    
    while True:
        try:
            user_input = input("") # Raw input
            if user_input.lower() in ["exit", "quit"]:
                running = False
                break
            
            # Inject user matter into the simulation
            engine.inject(user_input)
            
        except KeyboardInterrupt:
            running = False
            break
            
    print("\n🌌 Universe Heat Death. Goodbye.")

if __name__ == "__main__":
    run_wells()