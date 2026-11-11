from implementation.execution.experiment import ExperimentBase
from implementation.shared.execution.resources import Operations as ExecutionResourcesOperations
from implementation.shared.training.schedule import Operations as TrainingScheduleOperations
from implementation.shared.data.catalog import Operations as DataCatalogOperations
from implementation.methods.tempo_fl.data.partitions import Operations as DataPartitionsOperations
from implementation.shared.data.pipeline import Operations as DataPipelineOperations
from implementation.shared.models.mobilenet import Operations as ModelsMobilenetOperations
from implementation.methods.tempo_fl.evaluation.metrics import Operations as EvaluationMetricsOperations
from implementation.methods.tempo_fl.training.aggregation import Operations as TrainingAggregationOperations
from implementation.methods.tempo_fl.reporting.figures_tables import Operations as ReportingFiguresTablesOperations
from implementation.methods.tempo_fl.training.local import Operations as TrainingLocalOperations
from implementation.shared.persistence.weights import Operations as PersistenceWeightsOperations
from implementation.methods.tempo_fl.runtime.state import Operations as MethodsStateOperations
from implementation.methods.tempo_fl.inference.calibration import Operations as InferenceCalibrationOperations
from implementation.methods.tempo_fl.training.promotion import Operations as TrainingPromotionOperations
from implementation.methods.tempo_fl.data.scenarios import Operations as DataScenariosOperations
from implementation.shared.training.generic_support import Operations as TrainingGenericSupportOperations
from implementation.methods.tempo_fl.inference.routing import Operations as InferenceRoutingOperations
from implementation.shared.reporting.tflite import Operations as ReportingTfliteOperations
from implementation.methods.tempo_fl.reporting.dataset import Operations as ReportingDatasetTempoOperations
from implementation.methods.tempo_fl.reporting.splits import Operations as ReportingSplitsTempoOperations
from implementation.methods.tempo_fl.reporting.partitions import Operations as ReportingPartitionsTempoOperations
from implementation.methods.tempo_fl.training.rounds import Operations as MethodsRoundsTempoOperations
from implementation.methods.tempo_fl.reporting.history import Operations as ReportingHistoryTempoOperations
from implementation.methods.tempo_fl.reporting.final import Operations as ReportingFinalTempoOperations
from implementation.methods.tempo_fl.reporting.deployment import Operations as ReportingDeploymentTempoOperations
from implementation.methods.tempo_fl.training.prepare_generic import (
    Operations as TrainingPrepareGenericTempoOperations,
)
from implementation.methods.tempo_fl.training.evaluate_generic_clients import (
    Operations as TrainingEvaluateGenericClientsTempoOperations,
)
from implementation.methods.tempo_fl.training.specialize_clients import (
    Operations as TrainingSpecializeClientsTempoOperations,
)
from implementation.methods.tempo_fl.training.bootstrap_hierarchy import (
    Operations as TrainingBootstrapHierarchyTempoOperations,
)


class Experiment(
    ExecutionResourcesOperations,
    TrainingScheduleOperations,
    DataCatalogOperations,
    DataPartitionsOperations,
    DataPipelineOperations,
    ModelsMobilenetOperations,
    EvaluationMetricsOperations,
    TrainingAggregationOperations,
    ReportingFiguresTablesOperations,
    TrainingLocalOperations,
    PersistenceWeightsOperations,
    MethodsStateOperations,
    InferenceCalibrationOperations,
    TrainingPromotionOperations,
    DataScenariosOperations,
    TrainingGenericSupportOperations,
    InferenceRoutingOperations,
    ReportingTfliteOperations,
    ReportingDatasetTempoOperations,
    ReportingSplitsTempoOperations,
    ReportingPartitionsTempoOperations,
    MethodsRoundsTempoOperations,
    ReportingHistoryTempoOperations,
    ReportingFinalTempoOperations,
    ReportingDeploymentTempoOperations,
    TrainingPrepareGenericTempoOperations,
    TrainingEvaluateGenericClientsTempoOperations,
    TrainingSpecializeClientsTempoOperations,
    TrainingBootstrapHierarchyTempoOperations,
    ExperimentBase,
):
    """tempo reference protocol with isolated state."""

    evaluate_general_stage_rows = (
        EvaluationMetricsOperations.evaluate_general_stage_rows__tempo_hfl_flat
    )
    evaluate_owned_stage_rows = EvaluationMetricsOperations.evaluate_owned_stage_rows__tempo_hfl
    write_v12_2_metric_workbook = (
        ReportingFiguresTablesOperations.write_v12_2_metric_workbook__tempo
    )
    fresh_runtime_state = MethodsStateOperations.fresh_runtime_state__tempo

    def initialize_from_scratch(self):
        self.stage("Prepare Generic")
        self.prepare_generic()
        self.stage("Evaluate Generic Clients")
        self.evaluate_generic_clients()
        self.stage("Specialize Clients")
        self.specialize_clients()
        self.stage("Bootstrap Hierarchy")
        self.bootstrap_hierarchy()
        state = self._init_state
        for name in list(vars(self)):
            if name.startswith("_init_"):
                delattr(self, name)
        return state
