#pragma once

#include "EvalParams.h"
#include "EvalT.h"
#include "GameState.h"

#include <valarray>

using VectorT = std::valarray<double>;

struct EvalWithGradient {
    EvalCalcT eval;
    VectorT gradient;
};

using PstMapping = std::array<std::int8_t, kSquares>;

class Evaluator {
  public:
    struct EvalCalcParams : EvalParams {
        EvalCalcParams(const EvalParams& evalParams);

        EvalCalcT maxPhaseMaterial_{};
    };

    Evaluator();
    Evaluator(const EvalParams& params);

    [[nodiscard]] int getPieceSquareValue(Piece piece, BoardPosition position, Side side) const;

    [[nodiscard]] EvalCalcT evaluateRaw(const GameState& gameState) const;

    [[nodiscard]] EvalWithGradient evaluateWithGradient(const GameState& gameState) const;

    [[nodiscard]] EvalT evaluate(const GameState& gameState) const;

  private:
    EvalCalcParams params_;
};

[[nodiscard]] int getStaticPieceValue(Piece piece);

[[nodiscard]] bool isInsufficientMaterial(const GameState& gameState);

[[nodiscard]] EvalT evaluateNoLegalMoves(const GameState& gameState);

void getPstMapping(
        const GameState& gameState,
        const PstMapping*& whiteMapping,
        const PstMapping*& whiteKingMapping,
        const PstMapping*& blackMapping,
        const PstMapping*& blackKingMapping);
