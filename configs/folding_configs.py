"""
Folding parameters for the four design-space-exploration configurations.

Project 335 — Real-Time Quantized CNN Accelerator for Edge AI
Model: CNV-w1a1 (FINN / Brevitas), target PYNQ-Z1 @ 100 MHz

Only the MVAU layers that were actually tuned are listed. Every other layer
keeps the folding FINN assigns by default; these three are the ones that took
turns being the pipeline bottleneck.

How the cycle counts work
-------------------------
For an MVAU with weight matrix MW x MH:

    cycles = (MW / SIMD) * (MH / PE) * (output spatial size)

SIMD divides MW, PE divides MH, so only divisors are legal values. That
constraint is what decided each step below — we did not pick round numbers,
we picked the next legal divisor of the bottleneck layer.

Layer dimensions
----------------
    MVAU_hls_0 : MW = 27,  MH = 64    (first conv, 3x3x3 RGB input)
    MVAU_hls_6 : MW = 256, MH = 512
    MVAU_hls_7 : MW = 512, MH = 512
"""

# Configuration A — FINN's default folding. Baseline.
# Bottleneck: MVAU_hls_6 at 32,768 cycles  -> (256/4) * (512/1)
CONFIG_A = {
    "MVAU_hls_0": {"PE": 16, "SIMD": 3},
    "MVAU_hls_6": {"PE": 1,  "SIMD": 4},
    "MVAU_hls_7": {"PE": 1,  "SIMD": 8},
}

# Configuration B — widen the bottleneck: MVAU_hls_6 SIMD 4 -> 8.
# MVAU_hls_6 drops to 16,384 cycles; bottleneck moves to MVAU_hls_7 (32,768).
CONFIG_B = {
    "MVAU_hls_0": {"PE": 16, "SIMD": 3},
    "MVAU_hls_6": {"PE": 1,  "SIMD": 8},
    "MVAU_hls_7": {"PE": 1,  "SIMD": 8},
}

# Configuration C — widen the new bottleneck: MVAU_hls_7 SIMD 8 -> 16.
# MVAU_hls_7 drops to 16,384; bottleneck moves to MVAU_hls_0 (32,400).
# RECOMMENDED: 2416.88 FPS, 1.033 ms, +0.340 ns slack, 0 DSP.
CONFIG_C = {
    "MVAU_hls_0": {"PE": 16, "SIMD": 3},
    "MVAU_hls_6": {"PE": 1,  "SIMD": 8},
    "MVAU_hls_7": {"PE": 1,  "SIMD": 16},
}

# Configuration D — widen MVAU_hls_0: PE 16 -> 32.
# MVAU_hls_0 drops to 16,200; bottleneck moves to MVAU_hls_3 (28,800).
# Fastest, but costs ~1.3k LUTs and cuts timing slack to +0.082 ns.
CONFIG_D = {
    "MVAU_hls_0": {"PE": 32, "SIMD": 3},
    "MVAU_hls_6": {"PE": 1,  "SIMD": 8},
    "MVAU_hls_7": {"PE": 1,  "SIMD": 16},
}

CONFIGS = {"A": CONFIG_A, "B": CONFIG_B, "C": CONFIG_C, "D": CONFIG_D}

# Measured results, for reference. RTLSIM + Vivado Out-of-Context synthesis.
# Full data: ../results/finn_A_D_final_results.csv
RESULTS = {
    #        bottleneck      RTLSIM   stable   lat_ms     LUT     FF  BRAM DSP    Fmax    WNS
    "A": ("MVAU_hls_6", 1725.6583, 2254.6440, 1.35960, 20466, 27932,  98,  0, 104.8328, 0.461),
    "B": ("MVAU_hls_7", 2324.9920, 3220.6119, 1.19609, 20499, 27983,  98,  0, 109.7695, 0.890),
    "C": ("MVAU_hls_0", 2416.8834, 3220.6119, 1.03256, 20607, 28014,  98,  0, 103.5197, 0.340),
    "D": ("MVAU_hls_3", 2488.6765, 3329.4711, 1.01472, 21886, 29122,  99,  0, 100.8268, 0.082),
}
