# Real-Time Quantized CNN Accelerator for Edge AI

**Final project in Computer Engineering — Project 335**
Faculty of Engineering, Bar-Ilan University

Authors: Roni Volshtein, Kinanah Hanif
Academic supervisor: Dr. Leonid Yavits
Project mentor: David Freud
Track: Hardware Design

---

## What this project does

This project builds a hardware accelerator for quantized CNN inference on an
AMD/Xilinx **PYNQ-Z1** FPGA, using the **FINN** framework.

A pretrained **CNV-w1a1** model — a VGG-style CNN with 6 convolutional and
3 fully connected layers, trained on CIFAR-10 with 1-bit weights and
activations via Brevitas quantization-aware training — is compiled into a
**streaming dataflow** hardware architecture. Every network layer becomes its
own hardware block, and activations stream between blocks through on-chip
FIFOs instead of going out to DRAM.

We then ran a **design space exploration** over the folding parameters
(PE and SIMD) across four configurations, A through D, repeatedly locating the
slowest pipeline stage and widening it.

## Scope — please read

All results in this repository come from **cycle-accurate RTL simulation
(PyVerilator RTLSIM)** and **Vivado Out-of-Context synthesis**.

The design was **not deployed to a physical PYNQ-Z1 board**. There are no
on-board measurements, no bitstream, no power figures, and no GPU baseline.
The CPU baseline is a real measurement; the FPGA figures are simulated.
Speedup ratios below compare a measured CPU against simulated FPGA timing and
should be read in that light.

---

## Results

### Design space exploration (A → D)

| Config | Change made | Bottleneck | RTLSIM FPS | Stable FPS | Latency (ms) | LUT | FF | BRAM | DSP | Fmax (MHz) | WNS (ns) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (baseline) | default FINN folding | MVAU_hls_6 | 1725.66 | 2254.64 | 1.3596 | 20,466 | 27,932 | 98 | 0 | 104.83 | +0.461 |
| B | MVAU_hls_6 SIMD 4 → 8 | MVAU_hls_7 | 2324.99 | 3220.61 | 1.1961 | 20,499 | 27,983 | 98 | 0 | 109.77 | +0.890 |
| **C (recommended)** | MVAU_hls_7 SIMD 8 → 16 | MVAU_hls_0 | **2416.88** | 3220.61 | **1.0326** | 20,607 | 28,014 | 98 | 0 | 103.52 | **+0.340** |
| D (maximum) | MVAU_hls_0 PE 16 → 32 | MVAU_hls_3 | 2488.68 | 3329.47 | 1.0147 | 21,886 | 29,122 | 99 | 0 | 100.83 | +0.082 |

Latency is `latency_cycles` at a 10 ns clock period (100 MHz).
Source data: [`results/finn_A_D_final_results.csv`](results/finn_A_D_final_results.csv)

**Configuration C is the recommended design.** D is marginally faster (+3.0%
RTLSIM throughput) but costs 1,279 more LUTs and 1,108 more FFs, and drops the
timing margin from +0.340 ns to +0.082 ns. C keeps roughly four times the slack
for almost the same performance.

**Zero DSP slices are used in any configuration.** With 1-bit weights and
activations the multiply-accumulate collapses into XNOR plus popcount, which
maps onto LUTs rather than DSP blocks.

### CPU baseline vs. Configuration C

| Platform | Method | Throughput | Latency |
|---|---|---|---|
| Host CPU | PyTorch, 1 thread, batch = 1 | 42.63 ± 1.65 FPS | 23.49 ± 0.93 ms |
| FINN Config C | RTLSIM @ 100 MHz | 2416.88 FPS | 1.033 ms |
| Ratio | measured CPU vs. simulated FPGA | ≈ 56.7× | ≈ 22.7× |

CPU baseline: 5 runs × 500 inferences, batch size 1, after a 30-iteration
warm-up, timed with `time.perf_counter`.
Source data: [`results/cpu_vs_finn_C.csv`](results/cpu_vs_finn_C.csv)

### Functional verification

Configuration C was verified with `STITCHED_IP_RTLSIM` against the PyTorch
software model. Both returned class 3 — `Match = True`.

---

## Environment

| Component | Version |
|---|---|
| FINN | v0.10.x |
| Vivado / Vitis HLS | 2022.2 |
| Python | 3.10 |
| PyTorch | 1.13.1+cu116 |
| Target board | AMD/Xilinx PYNQ-Z1 |
| Clock target | 100 MHz (10 ns) |

FINN was run from its official Docker container. Vivado was mounted into the
container from the host.

> The exact FINN patch release and commit hash are not recorded in the notebook
> outputs. The build steps used (`step_hw_codegen`, `step_hw_ipgen`) are the
> FINN 0.10-series names, and the Python and PyTorch versions above match the
> FINN v0.10 image, so the environment is v0.10.x.

---

## Repository layout

```
notebooks/   Jupyter notebooks, with all outputs preserved
results/     Final measurement data (CSV) and the charts used in the book
hardware/    FINN-generated deployment packages and Vivado block diagrams
configs/     The folding parameters for configurations A–D
```

### Which notebook produces what

| Notebook | What it produces |
|---|---|
| `cnv_end2end_baseline.ipynb` | Configuration A — first full hardware build, identifies MVAU_hls_6 as the bottleneck |
| `cnv_folding_optimization.ipynb` | Configuration B — first SIMD widening |
| `cnv_folding_C.ipynb` | Configuration C — the recommended design |
| `cnv_folding_D.ipynb` | Configuration D — maximum throughput |
| `cnv_final_verification.ipynb` | `STITCHED_IP_RTLSIM` functional verification **and the CPU baseline benchmark** |
| `tfc_end2end_baseline.ipynb` | Fully-connected (TFC) reference flow, used while learning FINN |
| `tfc_end2end_verification_B.ipynb` | Verification of the TFC flow |

**Notebook outputs are deliberately not cleared.** The measured numbers reported
in the project book live in those outputs, and clearing them would remove the
evidence.

Note that `cnv_final_verification.ipynb` contains three CPU benchmark cells.
The figure used throughout the project book is the **5-run average — 42.63 FPS
and 23.49 ms** (cell 35). An earlier single-run cell reports 46.28 FPS; it is
kept for transparency but the 5-run average is the reported result.

### Hardware artifacts

`deploy-on-pynq-cnv.zip` and `deploy-on-pynq-tfc.zip` are the deployment
packages FINN generated for the two networks. `stitched_ip.png`, `top.pdf`,
`StreamingDataflowPartition_1.pdf` and `pynq_shell_project.png` are the
generated Vivado block designs.

---

## Reproducing

1. Pull the FINN v0.10.x Docker image and set `FINN_ROOT` to your FINN checkout.
2. Mount your Vivado 2022.2 installation into the container and point
   `VIVADO_PATH` at it.
3. Start Jupyter inside the container.
4. Run the notebooks in the order listed in the table above.

Exact figures depend on the FINN version, so a different release may produce
different cycle counts and resource numbers.

---

## Notes

Figures taken from the official FINN documentation are used in the project book
but are deliberately **not** redistributed here.

The full write-up is in the project book, *Real-Time Quantized CNN Accelerator
for Edge AI*, Project 335, Bar-Ilan University.
