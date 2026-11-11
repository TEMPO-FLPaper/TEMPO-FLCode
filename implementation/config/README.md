# Method configurations

This is the project's single configuration directory. Each method has its own folder and two presets. Shared validation and help metadata remain at this directory's root.

| CLI method key | Folder | Reference preset | Reduced verification preset |
| --- | --- | --- | --- |
| `tempo` | `tempo_fl/` | [reference.json](tempo_fl/reference.json) | [verification.json](tempo_fl/verification.json) |
| `ditto` | `ditto/` | [reference.json](ditto/reference.json) | [verification.json](ditto/verification.json) |
| `hfl` | `hfl/` | [reference.json](hfl/reference.json) | [verification.json](hfl/verification.json) |
| `flat` | `flat/` | [reference.json](flat/reference.json) | [verification.json](flat/verification.json) |

The `tempo_fl` folder corresponds to the `tempo` command-line key and follows the naming of `implementation/methods/tempo_fl/`.

## Choosing a preset

**Reference** preserves the scientific defaults of the corresponding authoritative notebook, including its seed, rounds, model and training settings. Notebook-specific filesystem paths receive the existing local portability adaptations. Defaults come from cell 3 of each notebook; method identity, hashes and important cell locations are recorded in [reference_sources.json](reference_sources.json).

**Verification** is explicitly reduced: two rounds, smaller model/input dimensions and smaller training budgets. It executes the real method and does not replace Reference defaults. Its results are not paper reproduction, and its two rounds do not reach the later distribution-shift phases. Saved resolved configurations list departures from Reference.

Settings with similar names can have different roles across methods. TEMPO-FL's routing, monitoring and promotion settings do not apply to the baselines. Ditto retains its own personalized-learning parameters; HFL and Flat FedAvg retain their respective aggregation/training parameters.

## Using configurations

In the interactive terminal, select a method, open **Categorized configuration**, and choose **Restore Reference** or **Load reduced verification preset**. Use `?NUMBER` before editing, or `?` during an edit, for contextual parameter help. Each method keeps its own draft when switching methods.

Create a working configuration rather than editing a shipped preset:

```powershell
tempo-fl config reference --method tempo --output runs/my-tempo.json
tempo-fl config verification --method ditto --output runs/check-ditto.json
tempo-fl config help persistence_rounds --method tempo
tempo-fl config show --config runs/my-tempo.json
tempo-fl config validate --config runs/my-tempo.json
tempo-fl config set --config runs/my-tempo.json --set num_rounds=10 --output runs/my-tempo-10.json
tempo-fl run --config runs/my-tempo-10.json
```

`python -m implementation` can replace `tempo-fl`. The shipped presets can also be loaded directly, for example:

```powershell
tempo-fl config validate --config implementation/config/hfl/reference.json
tempo-fl run --config implementation/config/flat/verification.json
```

Relative data and output paths resolve against the project home, independently of the current working directory. `TEMPO_FL_HOME` can override that home for an installed copy. Every run exports its fully resolved configuration; resume requires matching configuration, method, data and implementation identities.

For Python use:

```python
from implementation.config.schema import RunConfig, preset_path

reference = RunConfig.defaults("tempo")
verification = RunConfig.load(preset_path("tempo", "verification"))
reference.validate()
verification.validate()
```

## Shared files

- `schema.py`: preset lookup, method-specific validation, bounds, parameter metadata, configuration serialization and departure tracking.
- `help_text.py`: additional contextual descriptions for settings.
- `parameter_sources.json`: implementation locations associated with each setting.
- `reference_sources.json`: authoritative notebook identities, source hashes, important cell indices and implementation modules.
- `__init__.py`: Python package marker.

The method subfolders are data-resource directories, not additional Python packages. Packaging includes all eight presets and this guide. Moving these files into folders does not change their contents or the scientific algorithms. See the [main README](../../README.md) for installation, method descriptions and reproducibility qualifications.
