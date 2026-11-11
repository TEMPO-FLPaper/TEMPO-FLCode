from implementation.execution.experiment import ExperimentBase
from implementation.shared.execution.resources import Operations as ExecutionResourcesOperations
from implementation.shared.training.schedule import Operations as TrainingScheduleOperations
from implementation.shared.data.catalog import Operations as DataCatalogOperations
from implementation.shared.data.partitions import Operations as DataPartitionsOperations
from implementation.shared.data.pipeline import Operations as DataPipelineOperations
from implementation.shared.models.mobilenet import Operations as ModelsMobilenetOperations
from implementation.methods.ditto.evaluation.metrics import Operations as EvaluationMetricsOperations
from implementation.shared.training.aggregation import Operations as TrainingAggregationOperations
from implementation.shared.reporting.figures_tables import Operations as ReportingFiguresTablesOperations
from implementation.methods.ditto.training.local import Operations as TrainingLocalOperations
from implementation.methods.ditto.runtime.state import Operations as MethodsStateOperations
from implementation.shared.persistence.weights import Operations as PersistenceWeightsOperations
from implementation.shared.training.generic_support import Operations as TrainingGenericSupportOperations
from implementation.shared.reporting.tflite import Operations as ReportingTfliteOperations
from implementation.methods.ditto.reporting.dataset import Operations as ReportingDatasetDittoOperations
from implementation.methods.ditto.reporting.splits import Operations as ReportingSplitsDittoOperations
from implementation.methods.ditto.reporting.partitions import Operations as ReportingPartitionsDittoOperations
from implementation.methods.ditto.training.rounds import Operations as MethodsRoundsDittoOperations
from implementation.methods.ditto.reporting.history import Operations as ReportingHistoryDittoOperations
from implementation.methods.ditto.reporting.final import Operations as ReportingFinalDittoOperations
from implementation.methods.ditto.reporting.deployment import Operations as ReportingDeploymentDittoOperations
from implementation.methods.ditto.training.prepare_generic import (
    Operations as TrainingPrepareGenericDittoOperations,
)
from implementation.methods.ditto.training.bootstrap_models import (
    Operations as TrainingBootstrapModelsDittoOperations,
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
    ReportingDatasetDittoOperations,
    ReportingSplitsDittoOperations,
    ReportingPartitionsDittoOperations,
    MethodsRoundsDittoOperations,
    ReportingHistoryDittoOperations,
    ReportingFinalDittoOperations,
    ReportingDeploymentDittoOperations,
    TrainingPrepareGenericDittoOperations,
    TrainingBootstrapModelsDittoOperations,
    ExperimentBase,
):
    """ditto reference protocol with isolated state."""

    evaluate_general_stage_rows = EvaluationMetricsOperations.evaluate_general_stage_rows__ditto
    evaluate_owned_stage_rows = EvaluationMetricsOperations.evaluate_owned_stage_rows__ditto
    write_v12_2_metric_workbook = (
        ReportingFiguresTablesOperations.write_v12_2_metric_workbook__ditto_hfl_flat
    )
    fresh_runtime_state = MethodsStateOperations.fresh_runtime_state__ditto

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
