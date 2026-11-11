"""Reference-derived persistence/weights. See README.md."""

from pathlib import Path

class Operations:
    # Reference cells (zero-based): {'tempo': 23, 'ditto': 22, 'hfl': 21, 'flat': 21}
    def save_weights_to_file(self, weights, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_model.set_weights(weights)
        self.eval_model.save_weights(file_path)

    # Reference cells (zero-based): {'tempo': 23, 'ditto': 22, 'hfl': 21, 'flat': 21}
    def load_weights_from_file(self, file_path):
        self.eval_model.load_weights(file_path)
        return [w.copy() for w in self.eval_model.get_weights()]
