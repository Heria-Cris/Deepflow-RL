# core/hardware.py
class Device:
    def __init__(
        self,
        device_id,
        name,
        device_type,
        fp16_tflops,
        mem_bandwidth_gb_s,
        max_memory_gb,
        runtime_reserved_memory_gb=0.0,
        compute_efficiency=0.50,
        memory_efficiency=0.80,
        pcie_latency_us=0.0,
    ):
        """
        更接近真实系统的硬件建模：
        - fp16_tflops / mem_bandwidth_gb_s 仍表示名义峰值
        - compute_efficiency / memory_efficiency 用于映射到有效可用性能
        - runtime_reserved_memory_gb 表示系统/框架/allocator 预留内存
        """
        self.device_id = device_id
        self.name = name
        self.device_type = device_type

        self.fp16_tflops = float(fp16_tflops)
        self.mem_bandwidth_gb_s = float(mem_bandwidth_gb_s)
        self.max_memory_gb = float(max_memory_gb)

        self.runtime_reserved_memory_gb = float(runtime_reserved_memory_gb)
        self.compute_efficiency = float(compute_efficiency)
        self.memory_efficiency = float(memory_efficiency)

        self.pcie_latency_s = float(pcie_latency_us) / 1e6

    @property
    def effective_tflops(self) -> float:
        return self.fp16_tflops * self.compute_efficiency

    @property
    def effective_mem_bandwidth_gb_s(self) -> float:
        return self.mem_bandwidth_gb_s * self.memory_efficiency

    @property
    def available_memory_gb(self) -> float:
        return max(0.0, self.max_memory_gb - self.runtime_reserved_memory_gb)

    def compute_time(self, flops_gflops: float) -> float:
        """
        Time = FLOPs / effective_TFLOPS
        flops_gflops: GFLOPs
        """
        if self.effective_tflops <= 1e-9:
            return 9999.0
        return flops_gflops / (self.effective_tflops * 1000.0)

    def memory_access_time(self, data_size_mb: float) -> float:
        """
        显存访问时间，基于有效带宽而不是峰值带宽。
        """
        if self.effective_mem_bandwidth_gb_s <= 1e-12:
            return 9999.0
        data_size_gb = data_size_mb / 1024.0
        return data_size_gb / self.effective_mem_bandwidth_gb_s

    def __repr__(self):
        return (
            f"[{self.device_type.upper()}] {self.name} "
            f"(TFLOPS_eff={self.effective_tflops:.2f}, "
            f"BW_eff={self.effective_mem_bandwidth_gb_s:.2f} GB/s, "
            f"AvailMem={self.available_memory_gb:.2f} GB)"
        )


class NetworkLink:
    def __init__(
        self,
        bandwidth_mbps,
        latency_ms,
        bandwidth_efficiency=0.80,
        serialization_overhead_ms=0.0,
    ):
        """
        更真实的网络链路建模：
        Total Time = propagation_latency + serialization_overhead + size / effective_bandwidth
        """
        self.bandwidth_mbps = float(bandwidth_mbps)
        self.latency_s = float(latency_ms) / 1000.0

        self.bandwidth_efficiency = float(bandwidth_efficiency)
        self.serialization_overhead_s = float(serialization_overhead_ms) / 1000.0

        raw_gb_s = (self.bandwidth_mbps / 8.0) / 1024.0
        self.bandwidth_gb_s = raw_gb_s * self.bandwidth_efficiency

    def estimate_comm_time(self, data_size_mb: float) -> float:
        if data_size_mb <= 0:
            return 0.0

        if self.bandwidth_gb_s <= 1e-12:
            transmission_time = 9999.0
        else:
            transmission_time = (data_size_mb / 1024.0) / self.bandwidth_gb_s

        return self.latency_s + self.serialization_overhead_s + transmission_time