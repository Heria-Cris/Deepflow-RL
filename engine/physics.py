# engine/physics.py
from core.hardware import Device, NetworkLink
from core.model_spec import TransformerLayerSpec


class PhysicsEngine:
    """
    物理仿真引擎：负责将抽象的模型层级操作转换为具体的物理时间开销。
    基于 Roofline Model 和 Latency-Bandwidth Model。
    """

    @staticmethod
    def estimate_layer_latency(layer: TransformerLayerSpec,
                               device: Device,
                               batch_size: int,
                               seq_len: int,
                               is_decoding: bool) -> float:
        """
        修正后的物理公式：始终考虑 Memory Wall。
        """
        # 1. 计算算力开销 (Compute Bound)
        flops = layer.get_flops(batch_size, seq_len, is_decoding)
        compute_time = device.compute_time(flops)

        # 2. 计算显存 I/O 开销 (Memory Bound)
        # 修正：无论是 Prefill 还是 Decode，只要权重读取时间 > 计算时间，就是 IO Bound
        # 这种情况在 Batch 小、Seq 短时非常常见
        mem_access_mb = layer.get_parameter_count() * 2 / (1024 ** 2)
        io_time = device.memory_access_time(mem_access_mb)

        # Roofline Model: 取最大值
        return max(compute_time, io_time)

    @staticmethod
    def estimate_transmission_latency(link: NetworkLink,
                                      data_size_mb: float) -> float:
        """
        估算网络传输延迟。
        """
        return link.estimate_comm_time(data_size_mb)