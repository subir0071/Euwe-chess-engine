#pragma once

#include "BoardConstants.h"

#include <array>
#include <string>

using EvalCalcT = float;

struct TaperedTerm {
    EvalCalcT early;
    EvalCalcT late;
};

using SquareTable       = std::array<TaperedTerm, kSquares>;
using PieceSquareTables = std::array<SquareTable, kNumPieceTypes>;

struct EvalParams {
    [[nodiscard]] static EvalParams getEmptyParams();
    [[nodiscard]] static EvalParams getDefaultParams();

    std::array<EvalCalcT, kNumPieceTypes> phaseMaterialValues;

    std::array<TaperedTerm, kNumPieceTypes> pieceValues;
    PieceSquareTables pieceSquareTablesWhite;

    std::array<TaperedTerm, 7> passedPawnBonus;
    TaperedTerm doubledPawnPenalty;
    TaperedTerm isolatedPawnPenalty;

    std::array<TaperedTerm, 9> bishopPawnSameColorAdjustment;
    std::array<TaperedTerm, 9> bishopEnemyPawnSameColorAdjustment;

    TaperedTerm bishopPairBonus;
    TaperedTerm knightPairBonus;
    TaperedTerm rookPairBonus;

    TaperedTerm rookSemiOpenFileBonus;
    TaperedTerm rookOpenFileBonus;

    std::array<TaperedTerm, 9> knightPawnAdjustment;
    std::array<TaperedTerm, 9> rookPawnAdjustment;
    std::array<TaperedTerm, 9> queenPawnAdjustment;

    TaperedTerm kingVirtualMobilityPenalty;

    std::array<TaperedTerm, kNumPieceTypes> mobilityBonus;

    static constexpr int kDoubledPawnTropismIdx        = kNumPieceTypes - 1;
    static constexpr int kIsolatedPawnTropismIdx       = kNumPieceTypes;
    static constexpr int kPassedPawnTropismIdx         = kNumPieceTypes + 1;
    static constexpr std::size_t kNumTropismPieceTypes = kNumPieceTypes + 2;

    std::array<TaperedTerm, kNumTropismPieceTypes> ownKingTropism;
    std::array<TaperedTerm, kNumTropismPieceTypes> enemyKingTropism;

    TaperedTerm tempoBonus;

    std::array<std::array<TaperedTerm, 3>, kNumPieceTypes - 1> attackDefendAdjustment;

    std::array<TaperedTerm, 10> controlNearEnemyKing;
    std::array<TaperedTerm, kNumPieceTypes> kingAttackWeight;
    std::array<TaperedTerm, 6> numKingAttackersAdjustment;

    std::array<EvalCalcT, 9> oppositeColoredBishopFactor;
    EvalCalcT singleMinorFactor;
    EvalCalcT twoKnightsFactor;
    EvalCalcT rookVsMinorFactor;
    EvalCalcT rookAndMinorVsRookFactor;

    std::array<TaperedTerm, kNumPieceTypes - 1> piecePinnedAdjustment;

    TaperedTerm kingOpenFileAdjustment;
    TaperedTerm kingFlankOpenFileAdjustment;

  private:
    EvalParams() = default;
};

static constexpr std::size_t kNumEvalParams = sizeof(EvalParams) / sizeof(EvalCalcT);

using EvalParamArray = std::array<EvalCalcT, kNumEvalParams>;
static_assert(sizeof(EvalParams) == sizeof(EvalParamArray));

[[nodiscard]] EvalParamArray evalParamsToArray(const EvalParams& params);

[[nodiscard]] EvalParams evalParamsFromArray(const EvalParamArray& array);

[[nodiscard]] std::string evalParamsToString(const EvalParams& params);
