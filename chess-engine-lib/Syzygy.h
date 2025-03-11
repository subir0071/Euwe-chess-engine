#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "EvalT.h"
#include "GameState.h"
#include "Move.h"

[[nodiscard]] bool initSyzygy(const std::string& syzygyDirs);
void tearDownSyzygy();

[[nodiscard]] bool syzygyPathIsValid(std::string_view syzygyDirs);

[[nodiscard]] bool canProbeSyzgyRoot(const GameState& gameState);

[[nodiscard]] std::vector<Move> getSyzygyRootMoves(const GameState& gameState);

[[nodiscard]] bool canProbeSyzgyWdl(const GameState& gameState);

[[nodiscard]] std::optional<EvalT> probeSyzygyWdl(const GameState& gameState);
