from implementation.execution.experiment import ExperimentBase
from implementation.shared.execution.resources import Operations as ExecutionResourcesOperations
from implementation.shared.training.schedule import Operations as TrainingScheduleOperations
from implementation.shared.data.catalog import Operations as DataCatalogOperations
from implementation.shared.data.partitions import Operations as DataPartitionsOperations
from implementation.shared.data.pipeline import Operations as DataPipelineOperations
from implementation.shared.models.mobilenet import Operations as ModelsMobilenetOperations
from implementation.methods.flat.evaluation.metrics import Operations as EvaluationMetricsOperations
from implementation.shared.training.aggregation import Operations as TrainingAggregationOperations
from implementation.shared.reporting.figures_tables import Operations as ReportingFiguresTablesOperations
from implementation.shared.training.local import Operations as TrainingLocalOperations
from implementation.methods.flat.runtime.state import Operations as MethodsStateOperations
from implementation.shared.persistence.weights import Operations as PersistenceWeightsOperations
from implementation.shared.training.generic_support import Operations as TrainingGenericSupportOperations
from implementation.shared.reporting.tflite import Operations as ReportingTfliteOperations
from implementation.methods.flat.reporting.dataset import Operations as ReportingDatasetFlatOperations
from implementation.methods.flat.reporting.splits import Operations as ReportingSplitsFlatOperations
from implementation.methods.flat.reporting.partitions import Operations as ReportingPartitionsFlatOperations
from implementation.methods.flat.training.rounds import Operations as MethodsRoundsFlatOperations
from implementation.methods.flat.reporting.history import Operations as ReportingHistoryFlatOperations
from implementation.methods.flat.reporting.final import Operations as ReportingFinalFlatOperations
from implementation.methods.flat.reporting.deployment import Operations as ReportingDeploymentFlatOperations
from implementation.methods.flat.training.prepare_generic import (
    Operations as TrainingPrepareGenericFlatOperations,
)
from implementation.methods.flat.training.bootstrap_models import (
    Operations as TrainingBootstrapModelsFlatOperations,
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
    MethodsStateOperations,
    PersistenceWeightsOperations,
    TrainingGenericSupportOperations,
    ReportingTfliteOperations,
    ReportingDatasetFlatOperations,
    ReportingSplitsFlatOperations,
    ReportingPartitionsFlatOperations,
    MethodsRoundsFlatOperations,
    ReportingHistoryFlatOperations,
    ReportingFinalFlatOperations,
    ReportingDeploymentFlatOperations,
    TrainingPrepareGenericFlatOperations,
    TrainingBootstrapModelsFlatOperations,
    ExperimentBase,
):
    """flat reference protocol with isolated state."""

    evaluate_general_stage_rows = (
        EvaluationMetricsOperations.evaluate_general_stage_rows__tempo_hfl_flat
    )
    evaluate_owned_stage_rows = EvaluationMetricsOperations.evaluate_owned_stage_rows__flat
    write_v12_2_metric_workbook = (
        ReportingFiguresTablesOperations.write_v12_2_metric_workbook__ditto_hfl_flat
    )
    fresh_runtime_state = MethodsStateOperations.fresh_runtime_state__flat

    def initialize_from_scratch(self):
        self.stage("Prepare Generic")
        self.prepare_generic()
        self.stage("Bootstrap Models")
        self.bootstrap_models()
        state = self._init_state
        for name in list(vars(self)):
            if name.startswith("_init_"):
                delattr(self, name)
        return state
