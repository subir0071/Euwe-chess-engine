#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "EvalT.h"
#include "GameState.h"
#include "Move.h"

void initSyzygy(const std::filesystem::path& syzygyDir);
void tearDownSyzygy();

[[nodiscard]] bool canProbeSyzgyRoot(const GameState& gameState);

[[nodiscard]] std::vector<Move> getSyzygyRootMoves(const GameState& gameState);

[[nodiscard]] bool canProbeSyzgyWdl(const GameState& gameState);

[[nodiscard]] EvalT probeSyzygyWdl(const GameState& gameState);
