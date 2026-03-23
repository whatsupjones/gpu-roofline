# The GPU Efficiency Gap: A Systematic Method for Detecting Invisible Waste

**Christopher D. Jones**

March 2026

---

## Abstract

**Background.** GPU operators manage billion-dollar infrastructure using monitoring tools that cannot observe efficiency loss. The industry benchmarks GPU performance against a static roofline: one peak number, measured once, under ideal conditions. In practice, GPU performance is a trajectory. Thermal throttling, tenant contention, memory lifecycle transitions, and fleet synchronization each degrade that trajectory independently and at different rates. Their combined effect is not additive but contextual, varying with workload, time, and infrastructure configuration. The standard monitoring tools, `nvidia-smi` and NVIDIA DCGM, were designed for single-device health checks in a single-tenant era. They report instantaneous device state. They cannot measure the dynamic behavior that determines actual operational performance.

**Methods.** We introduce the concept of dynamic roofline tension: the measured ratio between burst and sustained GPU performance under real operating conditions. This extends the static roofline model [1] with a temporal dimension. Using this framework, we formalize six categories of invisible efficiency loss spanning device-level thermal physics, virtualization partitioning, and fleet-level coordination. We employ a synthetic model calibrated against hardware-validated roofline measurements on H100 SXM [2] (2,905 GB/s HBM3, 59.1 TFLOPS FP32, validated at 87 to 100% of spec) and H200 systems. The simulation executes 120,000 deterministic trials (20,000 per category) with NVML-matched noise injection and design-appropriate statistical tests (Mann-Whitney U, Wilcoxon signed-rank, Holm-Bonferroni correction).

**Results.** All six waste categories produce statistically significant effects with large standardized effect sizes (Cohen's d/d_z from 0.73 to 8.55). `nvidia-smi` and DCGM detect 0% of waste events across all six categories. The `gpu-roofline` framework detects 56.5 to 100%. The six categories map to three operational responses: directly recoverable capacity (ghost allocations, straggler detection), decision support for infrastructure design (contention measurement, thermal SLA data), and risk prevention (oversubscription detection).

**Conclusions.** We provide strong experimental evidence that static performance monitoring is fundamentally insufficient for GPU infrastructure. The six waste categories identified in this study are invisible precisely because existing tools treat performance as a fixed ceiling rather than a moving target. Dynamic roofline tension is the prerequisite for observing GPU efficiency loss. We publish the complete protocol, predictions, and benchmark as an open challenge to the GPU monitoring community.

**Keywords:** GPU efficiency, dynamic roofline, tension ratio, observability gap, invisible waste, MIG, virtualization, fleet operations, distributed training

---

*Full study available at: [github.com/whatsupjones/gpu-roofline](https://github.com/whatsupjones/gpu-roofline)*

---

## References

[1] S. Williams, A. Waterman, and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65-76, 2009.

[2] NVIDIA, "NVIDIA H100 Tensor Core GPU Datasheet," 2023. Available: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
