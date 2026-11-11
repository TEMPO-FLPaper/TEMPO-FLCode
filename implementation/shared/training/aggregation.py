"""Reference-derived training/aggregation. See README.md."""

import numpy as np

class Operations:
    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def weights_size_bytes(self, weights):
        return int(sum((np.asarray(w).nbytes for w in weights)))

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def weight_delta_l2(self, new_weights, old_weights):
        total = 0.0
        for new_value, old_value in zip(new_weights, old_weights):
            difference = new_value.astype(np.float64) - old_value.astype(np.float64)
            total += np.sum(difference * difference)
        return float(np.sqrt(total))

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def weighted_average_weights(self, weight_sets, sample_counts):
        sample_counts = np.asarray(sample_counts, dtype=np.float64)
        if sample_counts.sum() <= 0:
            raise ValueError("The total aggregation weight must be positive.")
        coefficients = sample_counts / sample_counts.sum()
        averaged = []
        for tensors in zip(*weight_sets):
            accumulator = np.zeros_like(tensors[0], dtype=np.float64)
            for coefficient, tensor in zip(coefficients, tensors):
                accumulator += coefficient * tensor.astype(np.float64)
            averaged.append(accumulator.astype(tensors[0].dtype))
        return averaged
