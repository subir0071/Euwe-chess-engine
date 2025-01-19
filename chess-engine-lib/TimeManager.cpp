#include "TimeManager.h"

#include "Math.h"

#include <format>

namespace {

[[nodiscard]] bool timeIsUp(const std::chrono::high_resolution_clock::time_point deadLine) {
    return std::chrono::high_resolution_clock::now() >= deadLine;
}

[[nodiscard]] bool shouldInterrupt(
        const std::chrono::high_resolution_clock::time_point deadLine, int& interruptCheckCounter) {
    static constexpr int interruptCheckInterval = 32;

    if (interruptCheckCounter > 0) {
        --interruptCheckCounter;
        return false;
    }

    interruptCheckCounter = interruptCheckInterval;

    return timeIsUp(deadLine);
}

}  // namespace

TimeManager::TimeManager() : moveOverhead_(std::chrono::milliseconds(100)) {}

void TimeManager::setFrontEnd(IFrontEnd* frontEnd) {
    frontEnd_ = frontEnd;

    frontEnd_->addOption(FrontEndOption::createInteger(
            "move_overhead_ms", (int)moveOverhead_.count(), 0, 10'000, [this](int v) {
                moveOverhead_ = std::chrono::milliseconds(v);
            }));
}

bool TimeManager::shouldInterruptSearch(const std::uint64_t nodesSearched) const {
    MY_ASSERT(mode_ != TimeManagementMode::None);

    switch (mode_) {
        case TimeManagementMode::TimeControl: {
            return shouldInterrupt(hardDeadLine_, interruptCheckCounter_);
        }

        case TimeManagementMode::Infinite: {
            return false;
        }

        case TimeManagementMode::FixedTime: {
            return shouldInterrupt(hardDeadLine_, interruptCheckCounter_);
        }

        case TimeManagementMode::FixedDepth: {
            return false;
        }

        case TimeManagementMode::FixedNodes: {
            return nodesSearched >= nodesTarget_;
        }

        default: {
            UNREACHABLE;
        }
    }
}

bool TimeManager::shouldStopAfterFullPly(const int depth) const {
    MY_ASSERT(mode_ != TimeManagementMode::None);

    switch (mode_) {
        case TimeManagementMode::TimeControl: {
            return timeIsUp(softDeadLine_);
        }

        case TimeManagementMode::Infinite: {
            return false;
        }

        case TimeManagementMode::FixedTime: {
            return timeIsUp(softDeadLine_);
        }

        case TimeManagementMode::FixedDepth: {
            return depth >= depthTarget_;
        }

        case TimeManagementMode::FixedNodes: {
            return false;
        }

        default: {
            UNREACHABLE;
        }
    }
}

void TimeManager::didTbProbe() const {
    // Force time check on next call.
    interruptCheckCounter_ = 0;
}

void TimeManager::configureForTimeControl(
        const std::chrono::milliseconds timeLeft,
        const std::chrono::milliseconds increment,
        const int movesToGo,
        const GameState& gameState) {
    startTime_ = std::chrono::high_resolution_clock::now();

    const int expectedGameLength = 40;
    const int expectedMovesLeft =
            max(10, expectedGameLength - (int)gameState.getHalfMoveClock() / 2);
    const int expectedMovesToGo = min(expectedMovesLeft, movesToGo);

    // timeLeft includes the increment for the current move.
    const std::chrono::milliseconds totalTime  = timeLeft + increment * (expectedMovesToGo - 1);
    const std::chrono::milliseconds maxTime    = timeLeft * 8 / 10 - moveOverhead_;
    const std::chrono::milliseconds timeTarget = totalTime / expectedMovesToGo - moveOverhead_;

    const std::chrono::milliseconds hardTimeBudget = std::min(maxTime, timeTarget * 4 / 3);
    const std::chrono::milliseconds softTimeBudget = hardTimeBudget / 2;

    if (frontEnd_) {
        frontEnd_->reportDebugString(std::format(
                "Time budget: soft {} ms / hard {} ms",
                softTimeBudget.count(),
                hardTimeBudget.count()));
    }

    mode_         = TimeManagementMode::TimeControl;
    softDeadLine_ = startTime_ + softTimeBudget;
    hardDeadLine_ = startTime_ + hardTimeBudget;
}

void TimeManager::configureForInfiniteSearch() {
    startTime_ = std::chrono::high_resolution_clock::now();

    mode_ = TimeManagementMode::Infinite;
}

void TimeManager::configureForFixedTimeSearch(const std::chrono::milliseconds time) {
    startTime_ = std::chrono::high_resolution_clock::now();

    mode_         = TimeManagementMode::FixedTime;
    softDeadLine_ = startTime_ + time - moveOverhead_;
    hardDeadLine_ = softDeadLine_;
}

void TimeManager::configureForFixedDepthSearch(const int depth) {
    startTime_ = std::chrono::high_resolution_clock::now();

    mode_        = TimeManagementMode::FixedDepth;
    depthTarget_ = depth;
}

void TimeManager::configureForFixedNodesSearch(const std::uint64_t nodes) {
    startTime_ = std::chrono::high_resolution_clock::now();

    mode_        = TimeManagementMode::FixedNodes;
    nodesTarget_ = nodes;
}

std::chrono::milliseconds TimeManager::getTimeElapsed() const {
    const auto elapsed = std::chrono::high_resolution_clock::now() - startTime_;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
}
