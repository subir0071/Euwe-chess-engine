#pragma once

#include "ScoredPosition.h"

#include "chess-engine-lib/EvalParams.h"

#include <array>
#include <vector>

void optimize(
        std::array<double, kNumEvalParams>& paramsDouble,
        const std::vector<ScoredPosition>& scoredPositions,
        bool fixPhaseValues,
        bool fixScale);
