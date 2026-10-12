// The shared program value types every model's Program speaks in. Include it inside the model's
// program namespace, after declaring detail::DemotionPlan and detail::ReleasePlan.

using program::AbortResult;
using program::BindingReservation;
using program::CapturePreparation;
using program::CheckpointHandle;
using program::CheckpointMetadata;
using program::CheckpointSummary;
using program::CommitResult;
using program::CommitRowResult;
using program::ContextOperationKind;
using program::DiscardResult;
using program::ExecutionUnit;
using program::ExecutionUnitKind;
using program::FinishResult;
using program::PendingBatch;
using program::PhysicalUsageSnapshot;
using program::PrefillProgress;
using program::PrefixShortlistKey;
using program::ReplayProgress;
using program::SequenceHandle;
using program::SourceCandidate;
using ContextDemotion = program::ContextDemotionOf<detail::DemotionPlan>;
using ContextRelease  = program::ContextReleaseOf<detail::ReleasePlan>;
