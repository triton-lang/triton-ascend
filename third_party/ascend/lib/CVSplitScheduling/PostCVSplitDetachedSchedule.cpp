/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "ascend/include/CVSplitScheduling/PostCVSplitDetachedSchedule.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <utility>

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

static bool isBuildablePlan(const PostCVSplitSchedulePlan &plan) {
  return plan.status == PostCVSplitSchedulePlanStatus::ValidKnownCapacity ||
         plan.status == PostCVSplitSchedulePlanStatus::ValidUnknownCapacity;
}

static llvm::StringRef
statusName(PostCVSplitDetachedScheduleStatus status) {
  switch (status) {
  case PostCVSplitDetachedScheduleStatus::Ready:
    return "ready";
  case PostCVSplitDetachedScheduleStatus::PlanNotBuildable:
    return "plan-not-buildable";
  case PostCVSplitDetachedScheduleStatus::MissingSlotAssignment:
    return "missing-slot-assignment";
  case PostCVSplitDetachedScheduleStatus::MissingEvent:
    return "missing-event";
  case PostCVSplitDetachedScheduleStatus::EventContractMismatch:
    return "event-contract-mismatch";
  case PostCVSplitDetachedScheduleStatus::InvalidCommandOrder:
    return "invalid-command-order";
  case PostCVSplitDetachedScheduleStatus::LiveDepthExceeded:
    return "live-depth-exceeded";
  case PostCVSplitDetachedScheduleStatus::ReductionMismatch:
    return "reduction-mismatch";
  case PostCVSplitDetachedScheduleStatus::BackendMismatch:
    return "backend-mismatch";
  }
  return "unknown";
}

static llvm::StringRef sideName(PostCVSplitDetachedSide side) {
  return side == PostCVSplitDetachedSide::Cube ? "cube" : "vector";
}

static llvm::StringRef roleName(PostCVSplitLineageRole role) {
  switch (role) {
  case PostCVSplitLineageRole::Score:
    return "score";
  case PostCVSplitLineageRole::Probability:
    return "probability";
  case PostCVSplitLineageRole::Product:
    return "product";
  }
  return "unknown";
}

static llvm::StringRef eventKindName(PostCVSplitEventKind kind) {
  return kind == PostCVSplitEventKind::Forward ? "forward" : "release";
}

static llvm::StringRef
commandName(PostCVSplitDetachedCommandKind kind) {
  switch (kind) {
  case PostCVSplitDetachedCommandKind::ScoreMatmul:
    return "score-matmul";
  case PostCVSplitDetachedCommandKind::ScoreReleaseWait:
    return "score-release-wait";
  case PostCVSplitDetachedCommandKind::ScorePublish:
    return "score-publish";
  case PostCVSplitDetachedCommandKind::ProbabilityWait:
    return "probability-wait";
  case PostCVSplitDetachedCommandKind::ProductMatmul:
    return "product-matmul";
  case PostCVSplitDetachedCommandKind::ProductReleaseWait:
    return "product-release-wait";
  case PostCVSplitDetachedCommandKind::ProductPublish:
    return "product-publish";
  case PostCVSplitDetachedCommandKind::ScoreWait:
    return "score-wait";
  case PostCVSplitDetachedCommandKind::RowwiseSoftmax:
    return "rowwise-softmax";
  case PostCVSplitDetachedCommandKind::DirectNzPack:
    return "direct-nz-pack";
  case PostCVSplitDetachedCommandKind::ProbabilityPublish:
    return "probability-publish";
  case PostCVSplitDetachedCommandKind::ScoreRelease:
    return "score-release";
  case PostCVSplitDetachedCommandKind::ProductWait:
    return "product-wait";
  case PostCVSplitDetachedCommandKind::ProductAccumulate:
    return "product-accumulate";
  case PostCVSplitDetachedCommandKind::ProductRelease:
    return "product-release";
  case PostCVSplitDetachedCommandKind::AffineReduce:
    return "affine-reduce";
  }
  return "unknown";
}

static llvm::StringRef resourceName(PrincipalResource resource) {
  switch (resource) {
  case PrincipalResource::Matrix:
    return "matrix";
  case PrincipalResource::Fixpipe:
    return "fixpipe";
  case PrincipalResource::Vector:
    return "vector";
  case PrincipalResource::Mte1:
    return "mte1";
  case PrincipalResource::Mte2:
    return "mte2";
  case PrincipalResource::Mte3:
    return "mte3";
  case PrincipalResource::ScalarControl:
    return "scalar";
  }
  return "unknown";
}

static const PostCVSplitSlotAssignment *
findSlot(const PostCVSplitSchedulePlan &plan, PostCVSplitLineageRole role,
         unsigned lane) {
  auto slot = llvm::find_if(plan.slots, [&](const auto &candidate) {
    return candidate.role == role && candidate.lane == lane;
  });
  return slot == plan.slots.end() ? nullptr : &*slot;
}

static const PostCVSplitEventPlan *
findForwardEvent(const PostCVSplitSchedulePlan &plan,
                 PostCVSplitLineageRole role, unsigned lane) {
  auto event = llvm::find_if(plan.events, [&](const auto &candidate) {
    return candidate.role == role &&
           candidate.kind == PostCVSplitEventKind::Forward &&
           candidate.laneOrSlot == lane;
  });
  return event == plan.events.end() ? nullptr : &*event;
}

static const PostCVSplitEventPlan *
findReleaseEvent(const PostCVSplitSchedulePlan &plan,
                 PostCVSplitLineageRole role, unsigned slot) {
  auto event = llvm::find_if(plan.events, [&](const auto &candidate) {
    return candidate.role == role &&
           candidate.kind == PostCVSplitEventKind::Release &&
           candidate.laneOrSlot == slot;
  });
  return event == plan.events.end() ? nullptr : &*event;
}

static const PostCVSplitVectorLanePlan *
findVectorLane(const PostCVSplitSchedulePlan &plan, unsigned lane) {
  auto lanePlan = llvm::find_if(plan.vectorLanes, [&](const auto &candidate) {
    return candidate.lane == lane;
  });
  return lanePlan == plan.vectorLanes.end() ? nullptr : &*lanePlan;
}

static llvm::SmallVector<PostCVSplitDetachedCommand> &
commandsFor(PostCVSplitDetachedSchedule &schedule,
            PostCVSplitDetachedSide side) {
  return side == PostCVSplitDetachedSide::Cube ? schedule.cubeCommands
                                               : schedule.vectorCommands;
}

static PostCVSplitDetachedCommand &
appendWork(PostCVSplitDetachedSchedule &schedule,
           PostCVSplitDetachedSide side,
           PostCVSplitDetachedCommandKind kind,
           PostCVSplitLineageRole role, unsigned lane, unsigned slot,
           PrincipalResource resource) {
  PostCVSplitDetachedCommand command;
  command.side = side;
  command.kind = kind;
  command.role = role;
  command.lane = lane;
  command.slot = slot;
  command.resource = resource;
  commandsFor(schedule, side).push_back(command);
  return commandsFor(schedule, side).back();
}

static void appendEvent(PostCVSplitDetachedSchedule &schedule,
                        PostCVSplitDetachedSide side,
                        PostCVSplitDetachedCommandKind kind,
                        PostCVSplitLineageRole role, unsigned lane,
                        unsigned slot, const PostCVSplitEventPlan &event,
                        bool signals) {
  PostCVSplitDetachedCommand &command =
      appendWork(schedule, side, kind, role, lane, slot,
                 signals ? event.signalingResource : event.waitingResource);
  command.logicalFlagId = event.logicalFlagId;
  command.hasEvent = true;
  command.signalsEvent = signals;
}

static void clearCommands(PostCVSplitDetachedSchedule &schedule) {
  schedule.cubeCommands.clear();
  schedule.vectorCommands.clear();
  schedule.eventUses.clear();
  schedule.observedMaxScoreLive = 0;
  schedule.observedMaxProductLive = 0;
  schedule.reductionRootValue = 0;
  schedule.verified = false;
  schedule.publicationEligible = false;
  schedule.mutationPerformed = false;
}

static unsigned countCommands(
    const llvm::SmallVectorImpl<PostCVSplitDetachedCommand> &commands,
    PostCVSplitDetachedCommandKind kind, unsigned lane) {
  return llvm::count_if(commands, [&](const auto &command) {
    return command.kind == kind && command.lane == lane;
  });
}

static std::optional<unsigned> findCommandIndex(
    const llvm::SmallVectorImpl<PostCVSplitDetachedCommand> &commands,
    PostCVSplitDetachedCommandKind kind, unsigned lane) {
  for (auto [index, command] : llvm::enumerate(commands))
    if (command.kind == kind && command.lane == lane)
      return index;
  return std::nullopt;
}

static bool ordered(std::initializer_list<std::optional<unsigned>> indices) {
  std::optional<unsigned> previous;
  for (std::optional<unsigned> index : indices) {
    if (!index || (previous && *index <= *previous))
      return false;
    previous = index;
  }
  return true;
}

static const PostCVSplitEventPlan *
findEventByFlag(const PostCVSplitSchedulePlan &plan, unsigned flag) {
  auto event = llvm::find_if(plan.events, [&](const auto &candidate) {
    return candidate.logicalFlagId == flag;
  });
  return event == plan.events.end() ? nullptr : &*event;
}

static unsigned expectedEventUses(const PostCVSplitSchedulePlan &plan,
                                  const PostCVSplitEventPlan &event) {
  if (event.kind == PostCVSplitEventKind::Forward)
    return 1;
  return llvm::count_if(plan.slots, [&](const auto &slot) {
    return slot.role == event.role && slot.slot == event.laneOrSlot;
  });
}

static PostCVSplitDetachedScheduleStatus verifyEventContracts(
    const PostCVSplitSchedulePlan &plan,
    PostCVSplitDetachedSchedule &schedule) {
  auto commandsHaveValidEvents = [&](const auto &commands) {
    for (const PostCVSplitDetachedCommand &command : commands) {
      if (!command.hasEvent)
        continue;
      const PostCVSplitEventPlan *event =
          findEventByFlag(plan, command.logicalFlagId);
      if (!event || event->role != command.role)
        return false;
      PrincipalResource expected = command.signalsEvent
                                       ? event->signalingResource
                                       : event->waitingResource;
      if (command.resource != expected)
        return false;
    }
    return true;
  };
  if (!commandsHaveValidEvents(schedule.cubeCommands) ||
      !commandsHaveValidEvents(schedule.vectorCommands))
    return PostCVSplitDetachedScheduleStatus::EventContractMismatch;

  for (const PostCVSplitEventPlan &event : plan.events) {
    PostCVSplitDetachedEventUse use;
    use.logicalFlagId = event.logicalFlagId;
    use.role = event.role;
    use.kind = event.kind;
    use.expectedUses = expectedEventUses(plan, event);
    auto countUse = [&](const auto &commands) {
      for (const PostCVSplitDetachedCommand &command : commands) {
        if (!command.hasEvent || command.logicalFlagId != event.logicalFlagId)
          continue;
        if (command.signalsEvent)
          ++use.signalUses;
        else
          ++use.waitUses;
      }
    };
    countUse(schedule.cubeCommands);
    countUse(schedule.vectorCommands);
    use.paired = use.expectedUses != 0 &&
                 use.signalUses == use.expectedUses &&
                 use.waitUses == use.expectedUses;
    schedule.eventUses.push_back(use);
    if (!use.paired)
      return PostCVSplitDetachedScheduleStatus::EventContractMismatch;
  }
  return PostCVSplitDetachedScheduleStatus::Ready;
}

static PostCVSplitDetachedScheduleStatus verifyCommandOrder(
    const PostCVSplitSchedulePlan &plan,
    PostCVSplitDetachedSchedule &schedule) {
  const unsigned lanes = plan.recurrence.logicalLaneCount;
  for (unsigned lane = 0; lane < lanes; ++lane) {
    for (PostCVSplitDetachedCommandKind kind : {
             PostCVSplitDetachedCommandKind::ScoreMatmul,
             PostCVSplitDetachedCommandKind::ScoreReleaseWait,
             PostCVSplitDetachedCommandKind::ScorePublish,
             PostCVSplitDetachedCommandKind::ProbabilityWait,
             PostCVSplitDetachedCommandKind::ProductMatmul,
             PostCVSplitDetachedCommandKind::ProductReleaseWait,
             PostCVSplitDetachedCommandKind::ProductPublish})
      if (countCommands(schedule.cubeCommands, kind, lane) != 1)
        return PostCVSplitDetachedScheduleStatus::InvalidCommandOrder;
    for (PostCVSplitDetachedCommandKind kind : {
             PostCVSplitDetachedCommandKind::ScoreWait,
             PostCVSplitDetachedCommandKind::RowwiseSoftmax,
             PostCVSplitDetachedCommandKind::DirectNzPack,
             PostCVSplitDetachedCommandKind::ProbabilityPublish,
             PostCVSplitDetachedCommandKind::ScoreRelease,
             PostCVSplitDetachedCommandKind::ProductWait,
             PostCVSplitDetachedCommandKind::ProductAccumulate,
             PostCVSplitDetachedCommandKind::ProductRelease})
      if (countCommands(schedule.vectorCommands, kind, lane) != 1)
        return PostCVSplitDetachedScheduleStatus::InvalidCommandOrder;

    if (!ordered({findCommandIndex(schedule.cubeCommands,
                                   PostCVSplitDetachedCommandKind::ScoreMatmul,
                                   lane),
                  findCommandIndex(
                      schedule.cubeCommands,
                      PostCVSplitDetachedCommandKind::ScoreReleaseWait, lane),
                  findCommandIndex(schedule.cubeCommands,
                                   PostCVSplitDetachedCommandKind::ScorePublish,
                                   lane)}) ||
        !ordered({findCommandIndex(
                      schedule.cubeCommands,
                      PostCVSplitDetachedCommandKind::ProbabilityWait, lane),
                  findCommandIndex(
                      schedule.cubeCommands,
                      PostCVSplitDetachedCommandKind::ProductMatmul, lane),
                  findCommandIndex(
                      schedule.cubeCommands,
                      PostCVSplitDetachedCommandKind::ProductReleaseWait,
                      lane),
                  findCommandIndex(
                      schedule.cubeCommands,
                      PostCVSplitDetachedCommandKind::ProductPublish, lane)}) ||
        !ordered({findCommandIndex(schedule.vectorCommands,
                                   PostCVSplitDetachedCommandKind::ScoreWait,
                                   lane),
                  findCommandIndex(
                      schedule.vectorCommands,
                      PostCVSplitDetachedCommandKind::RowwiseSoftmax, lane),
                  findCommandIndex(
                      schedule.vectorCommands,
                      PostCVSplitDetachedCommandKind::DirectNzPack, lane),
                  findCommandIndex(
                      schedule.vectorCommands,
                      PostCVSplitDetachedCommandKind::ProbabilityPublish,
                      lane),
                  findCommandIndex(
                      schedule.vectorCommands,
                      PostCVSplitDetachedCommandKind::ScoreRelease, lane),
                  findCommandIndex(schedule.vectorCommands,
                                   PostCVSplitDetachedCommandKind::ProductWait,
                                   lane),
                  findCommandIndex(
                      schedule.vectorCommands,
                      PostCVSplitDetachedCommandKind::ProductAccumulate, lane),
                  findCommandIndex(
                      schedule.vectorCommands,
                      PostCVSplitDetachedCommandKind::ProductRelease, lane)}))
      return PostCVSplitDetachedScheduleStatus::InvalidCommandOrder;
  }
  return PostCVSplitDetachedScheduleStatus::Ready;
}

static PostCVSplitDetachedScheduleStatus verifyLiveDepths(
    const PostCVSplitSchedulePlan &plan,
    PostCVSplitDetachedSchedule &schedule) {
  unsigned scoreLive = 0;
  unsigned productLive = 0;
  llvm::SmallVector<bool> scoreComputed(plan.recurrence.logicalLaneCount,
                                        false);
  llvm::SmallVector<bool> scorePublished(plan.recurrence.logicalLaneCount,
                                         false);
  llvm::SmallVector<bool> productComputed(plan.recurrence.logicalLaneCount,
                                          false);
  llvm::SmallVector<bool> productPublished(plan.recurrence.logicalLaneCount,
                                           false);
  for (const PostCVSplitDetachedCommand &command : schedule.cubeCommands) {
    switch (command.kind) {
    case PostCVSplitDetachedCommandKind::ScoreMatmul:
      if (scoreComputed[command.lane])
        return PostCVSplitDetachedScheduleStatus::LiveDepthExceeded;
      scoreComputed[command.lane] = true;
      ++scoreLive;
      schedule.observedMaxScoreLive =
          std::max(schedule.observedMaxScoreLive, scoreLive);
      break;
    case PostCVSplitDetachedCommandKind::ScorePublish:
      if (!scoreComputed[command.lane] || scorePublished[command.lane] ||
          scoreLive == 0)
        return PostCVSplitDetachedScheduleStatus::LiveDepthExceeded;
      scorePublished[command.lane] = true;
      --scoreLive;
      break;
    case PostCVSplitDetachedCommandKind::ProductMatmul:
      if (productComputed[command.lane])
        return PostCVSplitDetachedScheduleStatus::LiveDepthExceeded;
      productComputed[command.lane] = true;
      ++productLive;
      schedule.observedMaxProductLive =
          std::max(schedule.observedMaxProductLive, productLive);
      break;
    case PostCVSplitDetachedCommandKind::ProductPublish:
      if (!productComputed[command.lane] || productPublished[command.lane] ||
          productLive == 0)
        return PostCVSplitDetachedScheduleStatus::LiveDepthExceeded;
      productPublished[command.lane] = true;
      --productLive;
      break;
    default:
      break;
    }
  }
  if (scoreLive != 0 || productLive != 0 ||
      schedule.observedMaxScoreLive > plan.scoreLiveDepth ||
      schedule.observedMaxProductLive > plan.productLiveDepth ||
      llvm::any_of(scorePublished, [](bool value) { return !value; }) ||
      llvm::any_of(productPublished, [](bool value) { return !value; }))
    return PostCVSplitDetachedScheduleStatus::LiveDepthExceeded;
  return PostCVSplitDetachedScheduleStatus::Ready;
}

static PostCVSplitDetachedScheduleStatus verifyGeometryAndReduction(
    const PostCVSplitSchedulePlan &plan,
    PostCVSplitDetachedSchedule &schedule) {
  const unsigned lanes = plan.recurrence.logicalLaneCount;
  for (unsigned lane = 0; lane < lanes; ++lane) {
    const PostCVSplitVectorLanePlan *lanePlan = findVectorLane(plan, lane);
    const PostCVSplitSlotAssignment *scoreSlot =
        findSlot(plan, PostCVSplitLineageRole::Score, lane);
    const PostCVSplitSlotAssignment *probabilitySlot =
        findSlot(plan, PostCVSplitLineageRole::Probability, lane);
    if (!lanePlan || !scoreSlot || !probabilitySlot)
      return PostCVSplitDetachedScheduleStatus::MissingSlotAssignment;
    auto softmax = llvm::find_if(schedule.vectorCommands, [&](const auto &cmd) {
      return cmd.kind == PostCVSplitDetachedCommandKind::RowwiseSoftmax &&
             cmd.lane == lane;
    });
    auto packing = llvm::find_if(schedule.vectorCommands, [&](const auto &cmd) {
      return cmd.kind == PostCVSplitDetachedCommandKind::DirectNzPack &&
             cmd.lane == lane;
    });
    if (softmax == schedule.vectorCommands.end() ||
        packing == schedule.vectorCommands.end() ||
        softmax->slot != scoreSlot->slot ||
        packing->slot != probabilitySlot->slot ||
        softmax->rows != lanePlan->rows ||
        softmax->chunkWidth != lanePlan->chunkWidth ||
        softmax->chunkCount != lanePlan->chunkCount ||
        packing->rows != lanePlan->rows ||
        packing->chunkWidth != lanePlan->chunkWidth ||
        packing->chunkCount != lanePlan->chunkCount ||
        !packing->directNzPacking)
      return PostCVSplitDetachedScheduleStatus::InvalidCommandOrder;
  }

  llvm::SmallVector<const PostCVSplitDetachedCommand *> reductions;
  for (const PostCVSplitDetachedCommand &command : schedule.vectorCommands)
    if (command.kind == PostCVSplitDetachedCommandKind::AffineReduce)
      reductions.push_back(&command);
  if (reductions.size() != plan.reductionSteps.size())
    return PostCVSplitDetachedScheduleStatus::ReductionMismatch;
  for (auto [command, step] : llvm::zip_equal(reductions,
                                               plan.reductionSteps))
    if (command->reductionLevel != step.level ||
        command->leftValue != step.leftValue ||
        command->rightValue != step.rightValue ||
        command->resultValue != step.resultValue)
      return PostCVSplitDetachedScheduleStatus::ReductionMismatch;
  if (plan.reductionSteps.empty() ||
      reductions.back()->resultValue != plan.reductionRootValue)
    return PostCVSplitDetachedScheduleStatus::ReductionMismatch;
  schedule.reductionRootValue = plan.reductionRootValue;
  return PostCVSplitDetachedScheduleStatus::Ready;
}

static PostCVSplitDetachedScheduleStatus
verifyDetachedSchedule(const PostCVSplitSchedulePlan &plan,
                       PostCVSplitDetachedSchedule &schedule) {
  if (plan.backend.vfMergeLevel != 1 ||
      !plan.backend.disableAutoBindSubBlock ||
      !plan.backend.enableGraphSync)
    return PostCVSplitDetachedScheduleStatus::BackendMismatch;
  if (schedule.cubeCommands.size() != 7 * plan.recurrence.logicalLaneCount ||
      schedule.vectorCommands.size() !=
          8 * plan.recurrence.logicalLaneCount +
              plan.reductionSteps.size())
    return PostCVSplitDetachedScheduleStatus::InvalidCommandOrder;
  PostCVSplitDetachedScheduleStatus status =
      verifyCommandOrder(plan, schedule);
  if (status != PostCVSplitDetachedScheduleStatus::Ready)
    return status;
  status = verifyLiveDepths(plan, schedule);
  if (status != PostCVSplitDetachedScheduleStatus::Ready)
    return status;
  status = verifyGeometryAndReduction(plan, schedule);
  if (status != PostCVSplitDetachedScheduleStatus::Ready)
    return status;
  return verifyEventContracts(plan, schedule);
}

} // namespace

PostCVSplitDetachedSchedule
buildPostCVSplitDetachedSchedule(const PostCVSplitSchedulePlan &plan) {
  PostCVSplitDetachedSchedule schedule;
  schedule.logicalLaneCount = plan.recurrence.logicalLaneCount;
  schedule.backend = plan.backend;
  if (!isBuildablePlan(plan))
    return schedule;

  auto reject = [&](PostCVSplitDetachedScheduleStatus status) {
    schedule.status = status;
    clearCommands(schedule);
    return std::move(schedule);
  };
  auto slot = [&](PostCVSplitLineageRole role, unsigned lane) {
    return findSlot(plan, role, lane);
  };

  auto appendScoreMatmul = [&](unsigned lane) {
    const PostCVSplitSlotAssignment *scoreSlot =
        slot(PostCVSplitLineageRole::Score, lane);
    if (!scoreSlot)
      return false;
    appendWork(schedule, PostCVSplitDetachedSide::Cube,
               PostCVSplitDetachedCommandKind::ScoreMatmul,
               PostCVSplitLineageRole::Score, lane, scoreSlot->slot,
               PrincipalResource::Matrix);
    return true;
  };
  auto appendScorePublication = [&](unsigned lane) {
    const PostCVSplitSlotAssignment *scoreSlot =
        slot(PostCVSplitLineageRole::Score, lane);
    if (!scoreSlot)
      return PostCVSplitDetachedScheduleStatus::MissingSlotAssignment;
    const PostCVSplitEventPlan *release = findReleaseEvent(
        plan, PostCVSplitLineageRole::Score, scoreSlot->slot);
    const PostCVSplitEventPlan *forward =
        findForwardEvent(plan, PostCVSplitLineageRole::Score, lane);
    if (!release || !forward)
      return PostCVSplitDetachedScheduleStatus::MissingEvent;
    appendEvent(schedule, PostCVSplitDetachedSide::Cube,
                PostCVSplitDetachedCommandKind::ScoreReleaseWait,
                PostCVSplitLineageRole::Score, lane, scoreSlot->slot, *release,
                false);
    appendEvent(schedule, PostCVSplitDetachedSide::Cube,
                PostCVSplitDetachedCommandKind::ScorePublish,
                PostCVSplitLineageRole::Score, lane, scoreSlot->slot, *forward,
                true);
    return PostCVSplitDetachedScheduleStatus::Ready;
  };

  const unsigned lanes = plan.recurrence.logicalLaneCount;
  for (unsigned lane = 0; lane < std::min(lanes, plan.scoreLiveDepth); ++lane)
    if (!appendScoreMatmul(lane))
      return reject(
          PostCVSplitDetachedScheduleStatus::MissingSlotAssignment);
  for (unsigned lane = 0; lane < lanes; ++lane) {
    PostCVSplitDetachedScheduleStatus status = appendScorePublication(lane);
    if (status != PostCVSplitDetachedScheduleStatus::Ready)
      return reject(status);
    const unsigned refillLane = lane + plan.scoreLiveDepth;
    if (refillLane < lanes && !appendScoreMatmul(refillLane))
      return reject(
          PostCVSplitDetachedScheduleStatus::MissingSlotAssignment);
  }

  auto appendProductPublication = [&](unsigned lane) {
    const PostCVSplitSlotAssignment *productSlot =
        slot(PostCVSplitLineageRole::Product, lane);
    if (!productSlot)
      return PostCVSplitDetachedScheduleStatus::MissingSlotAssignment;
    const PostCVSplitEventPlan *release = findReleaseEvent(
        plan, PostCVSplitLineageRole::Product, productSlot->slot);
    const PostCVSplitEventPlan *forward =
        findForwardEvent(plan, PostCVSplitLineageRole::Product, lane);
    if (!release || !forward)
      return PostCVSplitDetachedScheduleStatus::MissingEvent;
    appendEvent(schedule, PostCVSplitDetachedSide::Cube,
                PostCVSplitDetachedCommandKind::ProductReleaseWait,
                PostCVSplitLineageRole::Product, lane, productSlot->slot,
                *release, false);
    appendEvent(schedule, PostCVSplitDetachedSide::Cube,
                PostCVSplitDetachedCommandKind::ProductPublish,
                PostCVSplitLineageRole::Product, lane, productSlot->slot,
                *forward, true);
    return PostCVSplitDetachedScheduleStatus::Ready;
  };

  for (unsigned lane = 0; lane < lanes; ++lane) {
    const PostCVSplitSlotAssignment *probabilitySlot =
        slot(PostCVSplitLineageRole::Probability, lane);
    const PostCVSplitSlotAssignment *productSlot =
        slot(PostCVSplitLineageRole::Product, lane);
    if (!probabilitySlot || !productSlot)
      return reject(
          PostCVSplitDetachedScheduleStatus::MissingSlotAssignment);
    const PostCVSplitEventPlan *probability =
        findForwardEvent(plan, PostCVSplitLineageRole::Probability, lane);
    if (!probability)
      return reject(PostCVSplitDetachedScheduleStatus::MissingEvent);
    appendEvent(schedule, PostCVSplitDetachedSide::Cube,
                PostCVSplitDetachedCommandKind::ProbabilityWait,
                PostCVSplitLineageRole::Probability, lane,
                probabilitySlot->slot, *probability, false);
    appendWork(schedule, PostCVSplitDetachedSide::Cube,
               PostCVSplitDetachedCommandKind::ProductMatmul,
               PostCVSplitLineageRole::Product, lane, productSlot->slot,
               PrincipalResource::Matrix);
    if (lane + 1 >= plan.productLiveDepth) {
      const unsigned publishLane = lane + 1 - plan.productLiveDepth;
      PostCVSplitDetachedScheduleStatus status =
          appendProductPublication(publishLane);
      if (status != PostCVSplitDetachedScheduleStatus::Ready)
        return reject(status);
    }
  }
  const unsigned firstDeferredProduct =
      lanes - std::min(lanes, plan.productLiveDepth - 1);
  for (unsigned lane = firstDeferredProduct; lane < lanes; ++lane) {
    PostCVSplitDetachedScheduleStatus status =
        appendProductPublication(lane);
    if (status != PostCVSplitDetachedScheduleStatus::Ready)
      return reject(status);
  }

  for (unsigned lane = 0; lane < lanes; ++lane) {
    const PostCVSplitSlotAssignment *scoreSlot =
        slot(PostCVSplitLineageRole::Score, lane);
    const PostCVSplitSlotAssignment *probabilitySlot =
        slot(PostCVSplitLineageRole::Probability, lane);
    const PostCVSplitSlotAssignment *productSlot =
        slot(PostCVSplitLineageRole::Product, lane);
    const PostCVSplitVectorLanePlan *lanePlan = findVectorLane(plan, lane);
    if (!scoreSlot || !probabilitySlot || !productSlot || !lanePlan)
      return reject(
          PostCVSplitDetachedScheduleStatus::MissingSlotAssignment);
    const PostCVSplitEventPlan *scoreForward =
        findForwardEvent(plan, PostCVSplitLineageRole::Score, lane);
    const PostCVSplitEventPlan *scoreRelease = findReleaseEvent(
        plan, PostCVSplitLineageRole::Score, scoreSlot->slot);
    const PostCVSplitEventPlan *probabilityForward =
        findForwardEvent(plan, PostCVSplitLineageRole::Probability, lane);
    const PostCVSplitEventPlan *productForward =
        findForwardEvent(plan, PostCVSplitLineageRole::Product, lane);
    const PostCVSplitEventPlan *productRelease = findReleaseEvent(
        plan, PostCVSplitLineageRole::Product, productSlot->slot);
    if (!scoreForward || !scoreRelease || !probabilityForward ||
        !productForward || !productRelease)
      return reject(PostCVSplitDetachedScheduleStatus::MissingEvent);

    appendEvent(schedule, PostCVSplitDetachedSide::Vector,
                PostCVSplitDetachedCommandKind::ScoreWait,
                PostCVSplitLineageRole::Score, lane, scoreSlot->slot,
                *scoreForward, false);
    PostCVSplitDetachedCommand &softmax =
        appendWork(schedule, PostCVSplitDetachedSide::Vector,
                   PostCVSplitDetachedCommandKind::RowwiseSoftmax,
                   PostCVSplitLineageRole::Score, lane, scoreSlot->slot,
                   PrincipalResource::Vector);
    softmax.rows = lanePlan->rows;
    softmax.chunkWidth = lanePlan->chunkWidth;
    softmax.chunkCount = lanePlan->chunkCount;
    PostCVSplitDetachedCommand &packing =
        appendWork(schedule, PostCVSplitDetachedSide::Vector,
                   PostCVSplitDetachedCommandKind::DirectNzPack,
                   PostCVSplitLineageRole::Probability, lane,
                   probabilitySlot->slot, PrincipalResource::Vector);
    packing.rows = lanePlan->rows;
    packing.chunkWidth = lanePlan->chunkWidth;
    packing.chunkCount = lanePlan->chunkCount;
    packing.directNzPacking = lanePlan->directNzPacking;
    appendEvent(schedule, PostCVSplitDetachedSide::Vector,
                PostCVSplitDetachedCommandKind::ProbabilityPublish,
                PostCVSplitLineageRole::Probability, lane,
                probabilitySlot->slot, *probabilityForward, true);
    appendEvent(schedule, PostCVSplitDetachedSide::Vector,
                PostCVSplitDetachedCommandKind::ScoreRelease,
                PostCVSplitLineageRole::Score, lane, scoreSlot->slot,
                *scoreRelease, true);
    appendEvent(schedule, PostCVSplitDetachedSide::Vector,
                PostCVSplitDetachedCommandKind::ProductWait,
                PostCVSplitLineageRole::Product, lane, productSlot->slot,
                *productForward, false);
    appendWork(schedule, PostCVSplitDetachedSide::Vector,
               PostCVSplitDetachedCommandKind::ProductAccumulate,
               PostCVSplitLineageRole::Product, lane, productSlot->slot,
               PrincipalResource::Vector);
    appendEvent(schedule, PostCVSplitDetachedSide::Vector,
                PostCVSplitDetachedCommandKind::ProductRelease,
                PostCVSplitLineageRole::Product, lane, productSlot->slot,
                *productRelease, true);
  }

  for (const PostCVSplitReductionStep &step : plan.reductionSteps) {
    PostCVSplitDetachedCommand &reduction =
        appendWork(schedule, PostCVSplitDetachedSide::Vector,
                   PostCVSplitDetachedCommandKind::AffineReduce,
                   PostCVSplitLineageRole::Product, 0, 0,
                   PrincipalResource::Vector);
    reduction.hasRole = false;
    reduction.hasSlot = false;
    reduction.reductionLevel = step.level;
    reduction.leftValue = step.leftValue;
    reduction.rightValue = step.rightValue;
    reduction.resultValue = step.resultValue;
  }

  schedule.status = verifyDetachedSchedule(plan, schedule);
  if (schedule.status != PostCVSplitDetachedScheduleStatus::Ready)
    return reject(schedule.status);
  schedule.verified = true;
  schedule.publicationEligible = false;
  schedule.mutationPerformed = false;
  return schedule;
}

void logPostCVSplitDetachedSchedule(
    const PostCVSplitDetachedSchedule &schedule) {
  LLVM_DEBUG({
    llvm::dbgs() << "[cv-split] detached-schedule status="
                 << statusName(schedule.status)
                 << " lanes=" << schedule.logicalLaneCount
                 << " cube-commands=" << schedule.cubeCommands.size()
                 << " vector-commands=" << schedule.vectorCommands.size()
                 << " event-contracts=" << schedule.eventUses.size()
                 << " verified=" << (schedule.verified ? "yes" : "no")
                 << " publication-eligible="
                 << (schedule.publicationEligible ? "yes" : "no")
                 << " mutation=no\n";
    auto logCommands = [&](const auto &commands) {
      for (auto [ordinal, command] : llvm::enumerate(commands)) {
        llvm::dbgs() << "[cv-split] detached-command side="
                     << sideName(command.side) << " ordinal=" << ordinal
                     << " kind=" << commandName(command.kind) << " role=";
        if (command.hasRole)
          llvm::dbgs() << roleName(command.role);
        else
          llvm::dbgs() << "none";
        llvm::dbgs() << " lane=" << command.lane << " slot=";
        if (command.hasSlot)
          llvm::dbgs() << command.slot;
        else
          llvm::dbgs() << "none";
        llvm::dbgs() << " resource=" << resourceName(command.resource)
                     << " rows=" << command.rows
                     << " chunk-width=" << command.chunkWidth
                     << " chunks=" << command.chunkCount
                     << " direct-nz="
                     << (command.directNzPacking ? "yes" : "no")
                     << " event=";
        if (command.hasEvent)
          llvm::dbgs() << command.logicalFlagId;
        else
          llvm::dbgs() << "none";
        llvm::dbgs() << " event-action="
                     << (command.hasEvent
                             ? (command.signalsEvent ? "signal" : "wait")
                             : "none")
                     << "\n";
      }
    };
    logCommands(schedule.cubeCommands);
    logCommands(schedule.vectorCommands);
    for (const PostCVSplitDetachedEventUse &use : schedule.eventUses)
      llvm::dbgs() << "[cv-split] detached-event flag="
                   << use.logicalFlagId << " role=" << roleName(use.role)
                   << " kind=" << eventKindName(use.kind)
                   << " expected=" << use.expectedUses
                   << " signals=" << use.signalUses
                   << " waits=" << use.waitUses
                   << " paired=" << (use.paired ? "yes" : "no") << "\n";
    llvm::dbgs() << "[cv-split] detached-live-depth score-max="
                 << schedule.observedMaxScoreLive
                 << " product-max=" << schedule.observedMaxProductLive
                 << " reduction-root=" << schedule.reductionRootValue << "\n";
    llvm::dbgs() << "[cv-split] detached-backend auto-bind=off graph-sync=on"
                    " vf-merge="
                 << schedule.backend.vfMergeLevel
                 << " attribute-emitted=no publication=no\n";
  });
}

} // namespace mlir::triton::cv_split
