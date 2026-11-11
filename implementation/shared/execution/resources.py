"""Reference-derived execution/resources. See README.md."""

import numpy as np

import tensorflow as tf

class Operations:
    # Reference cells (zero-based): {'tempo': 1, 'ditto': 1, 'hfl': 1, 'flat': 1}
    def current_process_ram_gb(self):
        """
        Linux/Colab current resident RAM.
        Does not allocate extra arrays.
        """
        try:
            with open("/proc/self/status", "r") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        kb = float(line.split()[1])
                        return kb / 1024.0 / 1024.0
        except Exception:
            pass
        return np.nan

    # Reference cells (zero-based): {'tempo': 1, 'ditto': 1, 'hfl': 1, 'flat': 1}
    def current_gpu_memory_gb(self):
        if not self.GPU_DEVICES:
            return np.nan
        try:
            info = tf.config.experimental.get_memory_info("GPU:0")
            return float(info["current"] / 1024**3)
        except Exception:
            return np.nan

    # Reference cells (zero-based): {'tempo': 1, 'ditto': 1, 'hfl': 1, 'flat': 1}
    def print_memory_snapshot(self, prefix):
        ram_gb = self.current_process_ram_gb()
        gpu_gb = self.current_gpu_memory_gb()
        ram_text = f"{ram_gb:.2f} GB" if not np.isnan(ram_gb) else "NA"
        gpu_text = f"{gpu_gb:.2f} GB" if not np.isnan(gpu_gb) else "NA"
        self.log(f"{prefix} | ProcessRAM={ram_text} | GPUCurrent={gpu_text}")
