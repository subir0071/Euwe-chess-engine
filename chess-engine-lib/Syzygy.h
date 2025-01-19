#pragma once

#include <filesystem>
#include <optional>

#include "EvalT.h"
#include "GameState.h"

void initSyzygy(const std::filesystem::path& syzygyDir);
void tearDownSyzygy();

[[nodiscard]] bool canProbeSyzgyWdl(const GameState& gameState);

[[nodiscard]] EvalT probeSyzygyWdl(const GameState& gameState);
