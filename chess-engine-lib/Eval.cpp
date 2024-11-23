#include "Eval.h"

#include "BitBoard.h"
#include "Macros.h"
#include "Math.h"
#include "PawnMasks.h"
#include "PieceControl.h"

#include <array>
#include <optional>
#include <type_traits>
#include <utility>

namespace {

struct TaperedEvaluation {
    EvalCalcT early = 0;
    EvalCalcT late  = 0;
};

struct PiecePositionEvaluation {
    TaperedEvaluation material{};
    TaperedEvaluation position{};

    EvalCalcT phaseMaterial = 0;

    BitBoard control     = BitBoard::Empty;
    BitBoard kingAttack  = BitBoard::Empty;
    int numKingAttackers = 0;
};

VectorT zeros() {
    return std::valarray<double>((EvalCalcT)0, kNumEvalParams);
}

template <bool CalcJacobians>
using ParamGradient = std::conditional_t<CalcJacobians, VectorT, std::monostate>;

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE ParamGradient<CalcJacobians> zeroGradient() {
    if constexpr (CalcJacobians) {
        return zeros();
    } else {
        return std::monostate{};
    }
}

template <bool CalcJacobians>
struct TaperedEvaluationJacobians;

template <>
struct TaperedEvaluationJacobians<false> {};

template <>
struct TaperedEvaluationJacobians<true> {
    ParamGradient<true> early = zeroGradient<true>();
    ParamGradient<true> late  = zeroGradient<true>();
};

template <bool CalcJacobians>
struct PiecePositionEvaluationJacobians {
    TaperedEvaluationJacobians<CalcJacobians> material{};
    TaperedEvaluationJacobians<CalcJacobians> position{};

    ParamGradient<CalcJacobians> phaseMaterialJacobians = zeroGradient<CalcJacobians>();
};

SquareTable getReflectedSquareTable(const SquareTable& table) {
    SquareTable result{};

    for (int i = 0; i < kSquares; ++i) {
        const BoardPosition position          = (BoardPosition)i;
        const BoardPosition reflectedPosition = getVerticalReflection(position);

        result[i] = table[(int)reflectedPosition];
    }

    return result;
}

PieceSquareTables getReflectedPieceSquareTables(const PieceSquareTables& tables) {
    PieceSquareTables result{};

    for (int i = 0; i < kNumPieceTypes; ++i) {
        result[i] = getReflectedSquareTable(tables[i]);
    }

    return result;
}

FORCE_INLINE std::size_t getParamIndex(const EvalParams& params, const EvalCalcT& param) {
    return (std::size_t)((std::byte*)&param - (std::byte*)&params) / sizeof(EvalCalcT);
}

template <bool CalcJacobians>
FORCE_INLINE void updateTaperedTerm(
        const Evaluator::EvalCalcParams& params,
        const TaperedTerm& term,
        TaperedEvaluation& eval,
        TaperedEvaluationJacobians<CalcJacobians>& jacobians,
        const EvalCalcT weight) {
    eval.early += term.early * weight;
    eval.late += term.late * weight;

    if constexpr (CalcJacobians) {
        jacobians.early[getParamIndex(params, term.early)] += weight;
        jacobians.late[getParamIndex(params, term.late)] += weight;
    }
}

template <bool CalcJacobians>
FORCE_INLINE void updatePiecePositionEvaluation(
        const Evaluator::EvalCalcParams& params,
        const int pieceIdx,
        const BoardPosition position,
        const Side side,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {
    result.phaseMaterial += params.phaseMaterialValues[pieceIdx];
    if constexpr (CalcJacobians) {
        jacobians.phaseMaterialJacobians[getParamIndex(
                params, params.phaseMaterialValues[pieceIdx])] += 1;
    }

    updateTaperedTerm(params, params.pieceValues[pieceIdx], result.material, jacobians.material, 1);

    // Manual update for position because of piece square table index mapping.

    result.position.early += params.pieceSquareTables[(int)side][pieceIdx][(int)position].early;
    result.position.late += params.pieceSquareTables[(int)side][pieceIdx][(int)position].late;

    if constexpr (CalcJacobians) {
        int positionForPieceSquare = (int)position;
        if (side == Side::Black) {
            positionForPieceSquare = (int)getVerticalReflection(position);
        }

        jacobians.position.early[getParamIndex(
                params, params.pieceSquareTablesWhite[pieceIdx][positionForPieceSquare].early)] +=
                1;
        jacobians.position.late[getParamIndex(
                params, params.pieceSquareTablesWhite[pieceIdx][positionForPieceSquare].late)] += 1;
    }
}

template <bool CalcJacobians>
FORCE_INLINE void updateMobilityEvaluation(
        const Evaluator::EvalCalcParams& params,
        const Piece piece,
        const BoardPosition position,
        const BitBoard anyPiece,
        const BitBoard ownOccupancy,
        const BitBoard enemyKingArea,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {
    const BitBoard control    = getPieceControlledSquares(piece, position, anyPiece);
    const BitBoard kingAttack = control & enemyKingArea;

    result.control |= control;
    result.kingAttack |= kingAttack;

    const bool isAttacker = kingAttack != BitBoard::Empty;

    result.numKingAttackers += isAttacker;
    updateTaperedTerm(
            params,
            params.kingAttackWeight[(int)piece],
            result.position,
            jacobians.position,
            isAttacker);

    const int mobility = popCount(control & ~ownOccupancy);
    updateTaperedTerm(
            params,
            params.mobilityBonus[(int)piece],
            result.position,
            jacobians.position,
            mobility);
}

template <bool CalcJacobians>
FORCE_INLINE void updateForVirtualKingMobility(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BoardPosition kingPosition,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {

    const BitBoard ownOccupancy = side == gameState.getSideToMove()
                                        ? gameState.getOccupancy().ownPiece
                                        : gameState.getOccupancy().enemyPiece;

    // Consider all of our own pieces as blockers, but for the enemy pieces we only consider pawns.
    // This is to account for the fact that the other enemy pieces are likely mobile and so should
    // not be relied upon to protect the king from sliding attacks.
    const BitBoard blockers =
            ownOccupancy | gameState.getPieceBitBoard(nextSide(side), Piece::Pawn);

    // We're only interested in squares from which an enemy slider could attack the king, so we
    // exclude the occupied squares themselves.
    const BitBoard virtualKingControl =
            getPieceControlledSquares(Piece::Queen, kingPosition, blockers) & ~blockers;

    const int virtualKingMobility = popCount(virtualKingControl);

    updateTaperedTerm(
            params,
            params.kingVirtualMobilityPenalty,
            result.position,
            jacobians.position,
            -virtualKingMobility);
}

static constexpr auto kTropisms = []() {
    std::array<std::array<std::uint8_t, kSquares>, kSquares> tropisms{};
    for (int from = 0; from < kSquares; ++from) {
        const auto [fromFile, fromRank] = fileRankFromPosition((BoardPosition)from);
        for (int to = 0; to < kSquares; ++to) {
            const auto [toFile, toRank] = fileRankFromPosition((BoardPosition)to);
            const int distance = constexprAbs(fromFile - toFile) + constexprAbs(fromRank - toRank);
            tropisms[from][to] = (std::uint8_t)(14 - distance);
        }
    }
    return tropisms;
}();

[[nodiscard]] FORCE_INLINE EvalCalcT
getTropism(const BoardPosition aPos, const BoardPosition bPos) {
    return (EvalCalcT)kTropisms[(int)aPos][(int)bPos];
}

template <bool CalcJacobians>
FORCE_INLINE void updateForKingTropism(
        const Evaluator::EvalCalcParams& params,
        const BoardPosition ownKingPosition,
        const BoardPosition enemyKingPosition,
        const int pieceIdx,
        const BoardPosition position,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {
    const EvalCalcT ownTropism   = getTropism(ownKingPosition, position);
    const EvalCalcT enemyTropism = getTropism(enemyKingPosition, position);

    updateTaperedTerm(
            params,
            params.ownKingTropism[pieceIdx],
            result.position,
            jacobians.position,
            ownTropism);

    updateTaperedTerm(
            params,
            params.enemyKingTropism[pieceIdx],
            result.position,
            jacobians.position,
            enemyTropism);
}

template <bool CalcJacobians>
FORCE_INLINE void updateForKingTropism(
        const Evaluator::EvalCalcParams& params,
        const BoardPosition ownKingPosition,
        const BoardPosition enemyKingPosition,
        const Piece piece,
        const BoardPosition position,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {
    updateForKingTropism(
            params, ownKingPosition, enemyKingPosition, (int)piece, position, result, jacobians);
}

FORCE_INLINE BitBoard getPinBitBoard(
        const GameState& gameState,
        const BoardPosition kingPosition,
        const Side kingSide,
        const BitBoard anyPiece) {
    BitBoard allPins = BitBoard::Empty;

    const BitBoard enemyRooksOrQueens =
            gameState.getPieceBitBoard(nextSide(kingSide), Piece::Rook)
            | gameState.getPieceBitBoard(nextSide(kingSide), Piece::Queen);

    if (enemyRooksOrQueens != BitBoard::Empty) {
        const BitBoard rookXRayFromKing = getRookXRay(kingPosition, anyPiece);
        BitBoard xRayingRooks           = rookXRayFromKing & enemyRooksOrQueens;

        if (xRayingRooks != BitBoard::Empty) {
            const BitBoard rookAttackFromKing = getRookAttack(kingPosition, anyPiece);
            xRayingRooks &= ~rookAttackFromKing;

            while (xRayingRooks != BitBoard::Empty) {
                const BoardPosition pinningPiecePosition = popFirstSetPosition(xRayingRooks);

                const BitBoard pinningBitBoard =
                        rookAttackFromKing & getRookAttack(pinningPiecePosition, anyPiece);

                allPins |= pinningBitBoard;
            }
        }
    }

    const BitBoard enemyBishopsOrQueens =
            gameState.getPieceBitBoard(nextSide(kingSide), Piece::Bishop)
            | gameState.getPieceBitBoard(nextSide(kingSide), Piece::Queen);

    if (enemyBishopsOrQueens != BitBoard::Empty) {
        const BitBoard bishopXRayFromKing = getBishopXRay(kingPosition, anyPiece);
        BitBoard xRayingBishops           = bishopXRayFromKing & enemyBishopsOrQueens;

        if (xRayingBishops != BitBoard::Empty) {
            const BitBoard bishopAttackFromKing = getBishopAttack(kingPosition, anyPiece);
            xRayingBishops &= ~bishopAttackFromKing;

            while (xRayingBishops != BitBoard::Empty) {
                const BoardPosition pinningPiecePosition = popFirstSetPosition(xRayingBishops);

                const BitBoard pinningBitBoard =
                        bishopAttackFromKing & getBishopAttack(pinningPiecePosition, anyPiece);

                allPins |= pinningBitBoard;
            }
        }
    }

    return allPins;
}

template <bool CalcJacobians>
FORCE_INLINE void updateForPins(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BoardPosition ownKingPosition,
        const BitBoard anyPiece,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {
    BitBoard pinBitBoard;
    if (side == gameState.getSideToMove() && gameState.getPinBitBoardIfAvailable().has_value()) {
        pinBitBoard = *gameState.getPinBitBoardIfAvailable();
    } else {
        pinBitBoard = getPinBitBoard(gameState, ownKingPosition, side, anyPiece);
    }

    if (pinBitBoard == BitBoard::Empty) {
        return;
    }

    for (int pieceIdx = 0; pieceIdx < kNumPieceTypes - 1; ++pieceIdx) {
        const BitBoard& pieceBitBoard = gameState.getPieceBitBoard(side, (Piece)pieceIdx);
        const int numPinnedPieces     = popCount(pieceBitBoard & pinBitBoard);

        updateTaperedTerm(
                params,
                params.piecePinnedAdjustment[pieceIdx],
                result.position,
                jacobians.position,
                numPinnedPieces);
    }
}

template <bool CalcJacobians>
FORCE_INLINE void evaluatePiecePositionsForSide(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BoardPosition ownKingPosition,
        const BoardPosition enemyKingPosition,
        const BitBoard enemyKingArea,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {
    const BitBoard ownPawns   = gameState.getPieceBitBoard(side, Piece::Pawn);
    const BitBoard enemyPawns = gameState.getPieceBitBoard(nextSide(side), Piece::Pawn);
    const BitBoard anyPawn    = ownPawns | enemyPawns;

    const BitBoard ownOccupancy = side == gameState.getSideToMove()
                                        ? gameState.getOccupancy().ownPiece
                                        : gameState.getOccupancy().enemyPiece;

    const BitBoard anyPiece =
            gameState.getOccupancy().ownPiece | gameState.getOccupancy().enemyPiece;

    const int numOwnPawns = popCount(ownPawns);

    // Knights
    {
        BitBoard pieceBitBoard = gameState.getPieceBitBoard(side, Piece::Knight);

        const int numKnights = popCount(pieceBitBoard);
        if (numKnights >= 2) {
            updateTaperedTerm(
                    params, params.knightPairBonus, result.material, jacobians.material, 1);
        }

        while (pieceBitBoard != BitBoard::Empty) {
            const BoardPosition position = popFirstSetPosition(pieceBitBoard);

            updatePiecePositionEvaluation<CalcJacobians>(
                    params, (int)Piece::Knight, position, side, result, jacobians);

            updateForKingTropism(
                    params,
                    ownKingPosition,
                    enemyKingPosition,
                    Piece::Knight,
                    position,
                    result,
                    jacobians);

            updateTaperedTerm(
                    params,
                    params.knightPawnAdjustment[numOwnPawns],
                    result.material,
                    jacobians.material,
                    1);

            updateMobilityEvaluation<CalcJacobians>(
                    params,
                    Piece::Knight,
                    position,
                    anyPiece,
                    ownOccupancy,
                    enemyKingArea,
                    result,
                    jacobians);
        }
    }

    // Bishops
    {
        BitBoard pieceBitBoard = gameState.getPieceBitBoard(side, Piece::Bishop);

        const std::array<int, 2> ownPawnsPerSquareColor = {
                popCount(ownPawns & kDarkSquareBitBoard),
                popCount(ownPawns & kLightSquareBitBoard),
        };
        const std::array<int, 2> enemyPawnsPerSquareColor = {
                popCount(enemyPawns & kDarkSquareBitBoard),
                popCount(enemyPawns & kLightSquareBitBoard),
        };

        std::array<bool, 2> hasBishopOfColor = {false, false};

        while (pieceBitBoard != BitBoard::Empty) {
            const BoardPosition position = popFirstSetPosition(pieceBitBoard);

            updatePiecePositionEvaluation<CalcJacobians>(
                    params, (int)Piece::Bishop, position, side, result, jacobians);

            const int squareColor = getSquareColor(position);

            hasBishopOfColor[squareColor] = true;

            updateTaperedTerm(
                    params,
                    params.bishopPawnSameColorAdjustment[ownPawnsPerSquareColor[squareColor]],
                    result.position,
                    jacobians.position,
                    1);

            updateTaperedTerm(
                    params,
                    params.bishopEnemyPawnSameColorAdjustment
                            [enemyPawnsPerSquareColor[squareColor]],
                    result.position,
                    jacobians.position,
                    1);

            updateForKingTropism(
                    params,
                    ownKingPosition,
                    enemyKingPosition,
                    Piece::Bishop,
                    position,
                    result,
                    jacobians);

            updateMobilityEvaluation<CalcJacobians>(
                    params,
                    Piece::Bishop,
                    position,
                    anyPiece,
                    ownOccupancy,
                    enemyKingArea,
                    result,
                    jacobians);
        }

        if (hasBishopOfColor[0] && hasBishopOfColor[1]) {
            updateTaperedTerm(
                    params, params.bishopPairBonus, result.material, jacobians.material, 1);
        }
    }

    // Rooks
    {
        BitBoard pieceBitBoard = gameState.getPieceBitBoard(side, Piece::Rook);

        const int numRooks = popCount(pieceBitBoard);
        if (numRooks >= 2) {
            updateTaperedTerm(params, params.rookPairBonus, result.material, jacobians.material, 1);
        }

        while (pieceBitBoard != BitBoard::Empty) {
            const BoardPosition position = popFirstSetPosition(pieceBitBoard);

            updatePiecePositionEvaluation<CalcJacobians>(
                    params, (int)Piece::Rook, position, side, result, jacobians);

            const BitBoard fileBitBoard = getFileBitBoard(position);
            const bool blockedByOwnPawn = (ownPawns & fileBitBoard) != BitBoard::Empty;
            const bool blockedByAnyPawn = (anyPawn & fileBitBoard) != BitBoard::Empty;

            if (!blockedByAnyPawn) {
                updateTaperedTerm(
                        params, params.rookOpenFileBonus, result.position, jacobians.position, 1);
            } else if (!blockedByOwnPawn) {
                updateTaperedTerm(
                        params,
                        params.rookSemiOpenFileBonus,
                        result.position,
                        jacobians.position,
                        1);
            }

            updateForKingTropism(
                    params,
                    ownKingPosition,
                    enemyKingPosition,
                    Piece::Rook,
                    position,
                    result,
                    jacobians);

            updateTaperedTerm(
                    params,
                    params.rookPawnAdjustment[numOwnPawns],
                    result.material,
                    jacobians.material,
                    1);

            updateMobilityEvaluation<CalcJacobians>(
                    params,
                    Piece::Rook,
                    position,
                    anyPiece,
                    ownOccupancy,
                    enemyKingArea,
                    result,
                    jacobians);
        }
    }

    // Queens
    {
        BitBoard pieceBitBoard = gameState.getPieceBitBoard(side, Piece::Queen);

        while (pieceBitBoard != BitBoard::Empty) {
            const BoardPosition position = popFirstSetPosition(pieceBitBoard);

            updatePiecePositionEvaluation<CalcJacobians>(
                    params, (int)Piece::Queen, position, side, result, jacobians);

            updateForKingTropism(
                    params,
                    ownKingPosition,
                    enemyKingPosition,
                    Piece::Queen,
                    position,
                    result,
                    jacobians);

            updateTaperedTerm(
                    params,
                    params.queenPawnAdjustment[numOwnPawns],
                    result.material,
                    jacobians.material,
                    1);

            updateMobilityEvaluation<CalcJacobians>(
                    params,
                    Piece::Queen,
                    position,
                    anyPiece,
                    ownOccupancy,
                    enemyKingArea,
                    result,
                    jacobians);
        }
    }

    // King
    {
        updatePiecePositionEvaluation<CalcJacobians>(
                params, (int)Piece::King, ownKingPosition, side, result, jacobians);

        // no mobility bonus for king

        updateForVirtualKingMobility<CalcJacobians>(
                params, gameState, side, ownKingPosition, result, jacobians);

        // Note king attack data between kings was calculated during initialization.

        // King safety. Note: we rely on the king being evaluated last, so that king attack data is complete.
        const int controlNearEnemyKing = popCount(result.kingAttack);

        updateTaperedTerm(
                params,
                params.controlNearEnemyKing[controlNearEnemyKing],
                result.position,
                jacobians.position,
                1);

        const int numKingAttackersIdx =
                min(result.numKingAttackers, (int)params.numKingAttackersAdjustment.size() - 1);
        updateTaperedTerm(
                params,
                params.numKingAttackersAdjustment[numKingAttackersIdx],
                result.position,
                jacobians.position,
                1);

        updateForPins(params, gameState, side, ownKingPosition, anyPiece, result, jacobians);
    }
}

template <bool CalcJacobians>
FORCE_INLINE void evaluateAttackDefend(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BitBoard ownControl,
        const BitBoard enemyControl,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {
    const std::array<BitBoard, 3> attackDefendBitBoards = {
            ownControl & ~enemyControl,  // defended, not attacked
            ~ownControl & enemyControl,  // not defended, attacked (hanging)
            ownControl & enemyControl,   // defended, attacked
    };

    // Skip the king: defending the king is useless, and the king should never be under attack
    // (i.e., in check) when running eval.
    for (int pieceIdx = 0; pieceIdx < kNumPieceTypes - 1; ++pieceIdx) {
        const BitBoard pieceBitBoard = gameState.getPieceBitBoard(side, (Piece)pieceIdx);

        for (int attackDefendIdx = 0; attackDefendIdx < 3; ++attackDefendIdx) {
            const BitBoard relevantPieces = pieceBitBoard & attackDefendBitBoards[attackDefendIdx];
            const int numRelevantPieces   = popCount(relevantPieces);

            updateTaperedTerm(
                    params,
                    params.attackDefendAdjustment[pieceIdx][attackDefendIdx],
                    result.position,
                    jacobians.position,
                    numRelevantPieces);
        }
    }
}

template <bool CalcJacobians>
FORCE_INLINE void modifyForFactor(
        const Evaluator::EvalCalcParams& params,
        const EvalCalcT& factor,
        EvalCalcT& whiteEval,
        ParamGradient<CalcJacobians>& whiteEvalGradient) {
    if constexpr (CalcJacobians) {
        whiteEvalGradient *= factor;

        auto& dEvalDFactor = whiteEvalGradient[getParamIndex(params, factor)];
        MY_ASSERT(dEvalDFactor == 0);
        dEvalDFactor = whiteEval;
    }

    whiteEval *= factor;
}

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE bool correctForOppositeColoredBishops(
        const GameState& gameState,
        const Evaluator::EvalCalcParams& params,
        EvalCalcT& whiteEval,
        ParamGradient<CalcJacobians>& whiteEvalGradient) {
    const BitBoard anyOtherPiece = gameState.getPieceBitBoard(Side::White, Piece::Knight)
                                 | gameState.getPieceBitBoard(Side::White, Piece::Rook)
                                 | gameState.getPieceBitBoard(Side::White, Piece::Queen)
                                 | gameState.getPieceBitBoard(Side::Black, Piece::Knight)
                                 | gameState.getPieceBitBoard(Side::Black, Piece::Rook)
                                 | gameState.getPieceBitBoard(Side::Black, Piece::Queen);

    if (anyOtherPiece != BitBoard::Empty) {
        return false;
    }

    const BitBoard whiteDarkBishops =
            gameState.getPieceBitBoard(Side::White, Piece::Bishop) & kDarkSquareBitBoard;
    const BitBoard whiteLightBishops =
            gameState.getPieceBitBoard(Side::White, Piece::Bishop) & kLightSquareBitBoard;
    const BitBoard blackDarkBishops =
            gameState.getPieceBitBoard(Side::Black, Piece::Bishop) & kDarkSquareBitBoard;
    const BitBoard blackLightBishops =
            gameState.getPieceBitBoard(Side::Black, Piece::Bishop) & kLightSquareBitBoard;

    const bool bishopsAreOppositeColor =
            ((whiteDarkBishops != BitBoard::Empty) ^ (blackDarkBishops != BitBoard::Empty))
            && ((whiteLightBishops != BitBoard::Empty) ^ (blackLightBishops != BitBoard::Empty));

    if (!bishopsAreOppositeColor) {
        return false;
    }

    const int whitePawns = popCount(gameState.getPieceBitBoard(Side::White, Piece::Pawn));
    const int blackPawns = popCount(gameState.getPieceBitBoard(Side::Black, Piece::Pawn));

    const int pawnDelta = std::abs(whitePawns - blackPawns);

    modifyForFactor<CalcJacobians>(
            params, params.oppositeColoredBishopFactor[pawnDelta], whiteEval, whiteEvalGradient);

    return true;
}

template <bool CalcJacobians>
FORCE_INLINE void correctForDrawish(
        const GameState& gameState,
        const Evaluator::EvalCalcParams& params,
        EvalCalcT& whiteEval,
        ParamGradient<CalcJacobians>& whiteEvalGradient) {
    if (correctForOppositeColoredBishops<CalcJacobians>(
                gameState, params, whiteEval, whiteEvalGradient)) {
        return;
    }

    const bool whiteIsStronger = whiteEval >= 0;

    const int strongerSidePawns =
            whiteIsStronger ? popCount(gameState.getPieceBitBoard(Side::White, Piece::Pawn))
                            : popCount(gameState.getPieceBitBoard(Side::Black, Piece::Pawn));

    if (strongerSidePawns > 0) {
        return;
    }

    int strongSideKnights = popCount(gameState.getPieceBitBoard(Side::White, Piece::Knight));
    int strongSideBishops = popCount(gameState.getPieceBitBoard(Side::White, Piece::Bishop));
    int strongSideRooks   = popCount(gameState.getPieceBitBoard(Side::White, Piece::Rook));
    int strongSideQueens  = popCount(gameState.getPieceBitBoard(Side::White, Piece::Queen));

    int weakSideKnights = popCount(gameState.getPieceBitBoard(Side::Black, Piece::Knight));
    int weakSideBishops = popCount(gameState.getPieceBitBoard(Side::Black, Piece::Bishop));
    int weakSideRooks   = popCount(gameState.getPieceBitBoard(Side::Black, Piece::Rook));
    int weakSideQueens  = popCount(gameState.getPieceBitBoard(Side::Black, Piece::Queen));

    if (!whiteIsStronger) {
        std::swap(strongSideKnights, weakSideKnights);
        std::swap(strongSideBishops, weakSideBishops);
        std::swap(strongSideRooks, weakSideRooks);
        std::swap(strongSideQueens, weakSideQueens);
    }

    const int strongSideMinorPieces = strongSideKnights + strongSideBishops;
    const int strongSideMajorPieces = strongSideRooks + strongSideQueens;

    const int weakSideMinorPieces = weakSideKnights + weakSideBishops;
    const int weakSideMajorPieces = weakSideRooks + weakSideQueens;

    // With only a single minor piece you can't reliably deliver mate.
    if (strongSideMajorPieces == 0 && strongSideMinorPieces == 1) {
        modifyForFactor<CalcJacobians>(
                params, params.singleMinorFactor, whiteEval, whiteEvalGradient);
        return;
    }

    // Only two knights; this is insufficient material once the weaker side has lost their material.
    if (strongSideMajorPieces == 0 && strongSideKnights == 2 && strongSideBishops == 0) {
        modifyForFactor<CalcJacobians>(
                params, params.twoKnightsFactor, whiteEval, whiteEvalGradient);
        return;
    }

    // Rook vs a minor piece is drawish.
    if (strongSideRooks == 1 && strongSideQueens == 0 && strongSideMinorPieces == 0
        && (weakSideMinorPieces == 1 && weakSideMajorPieces == 0)) {
        modifyForFactor<CalcJacobians>(
                params, params.rookVsMinorFactor, whiteEval, whiteEvalGradient);
        return;
    }

    // Rook and minor vs rook is drawish.
    if (strongSideRooks == 1 && strongSideQueens == 0 && strongSideMinorPieces == 1
        && weakSideRooks == 1 && weakSideQueens == 0 && weakSideMinorPieces == 0) {
        modifyForFactor<CalcJacobians>(
                params, params.rookAndMinorVsRookFactor, whiteEval, whiteEvalGradient);
        return;
    }
}

template <bool CalcJacobians>
FORCE_INLINE void evaluatePawnsForSide(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BoardPosition ownKingPosition,
        const BoardPosition enemyKingPosition,
        const BitBoard enemyKingArea,
        PiecePositionEvaluation& result,
        PiecePositionEvaluationJacobians<CalcJacobians>& jacobians) {
    const BitBoard ownPawns   = gameState.getPieceBitBoard(side, Piece::Pawn);
    const BitBoard enemyPawns = gameState.getPieceBitBoard(nextSide(side), Piece::Pawn);

    BitBoard pawnBitBoard = ownPawns;

    while (pawnBitBoard != BitBoard::Empty) {
        const BoardPosition position = popFirstSetPosition(pawnBitBoard);
        const auto [file, rank]      = fileRankFromPosition(position);

        updatePiecePositionEvaluation<CalcJacobians>(
                params, (int)Piece::Pawn, position, side, result, jacobians);

        const BitBoard passedPawnOpponentMask = getPassedPawnOpponentMask(position, side);
        const BitBoard forwardMask            = getPawnForwardMask(position, side);
        const BitBoard neighborMask           = getPawnNeighborFileMask(position);

        const BitBoard opponentBlockers = enemyPawns & passedPawnOpponentMask;
        const BitBoard ownBlockers      = ownPawns & forwardMask;
        const BitBoard ownNeighbors     = ownPawns & neighborMask;

        const bool isDoubledPawn = ownBlockers != BitBoard::Empty;
        const bool isPassedPawn  = !isDoubledPawn && opponentBlockers == BitBoard::Empty;
        const bool isIsolated    = ownNeighbors == BitBoard::Empty;

        int tropismIdx = (int)Piece::Pawn;

        if (isDoubledPawn) {
            updateTaperedTerm(
                    params, params.doubledPawnPenalty, result.position, jacobians.position, -1);

            tropismIdx = EvalParams::kDoubledPawnTropismIdx;
        } else if (isPassedPawn) {
            const int rank                = rankFromPosition(position);
            const int distanceToPromotion = side == Side::White ? kRanks - 1 - rank : rank;

            updateTaperedTerm(
                    params,
                    params.passedPawnBonus[distanceToPromotion],
                    result.position,
                    jacobians.position,
                    1);

            tropismIdx = EvalParams::kPassedPawnTropismIdx;
        } else if (isIsolated) {
            tropismIdx = EvalParams::kIsolatedPawnTropismIdx;
        }

        if (isIsolated) {
            updateTaperedTerm(
                    params, params.isolatedPawnPenalty, result.position, jacobians.position, -1);
        }

        updateForKingTropism(
                params,
                ownKingPosition,
                enemyKingPosition,
                tropismIdx,
                position,
                result,
                jacobians);
    }

    const BitBoard pawnControl = getPawnControlledSquares(ownPawns, side);
    result.control |= pawnControl;

    const BitBoard kingAttack = enemyKingArea & pawnControl;
    result.kingAttack |= kingAttack;

    const int numAttackedSquares = popCount(kingAttack);
    result.numKingAttackers += numAttackedSquares;

    updateTaperedTerm(
            params,
            params.kingAttackWeight[(int)Piece::Pawn],
            result.position,
            jacobians.position,
            numAttackedSquares);
}

[[nodiscard]] FORCE_INLINE ParamGradient<true> getMaxPhaseMaterialGradient(
        const EvalParams& params) {
    /*
    maxPhaseMaterial = 2 * 8 * evalParams.phaseMaterialValues[(int)Piece::Pawn]
                     + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Knight]
                     + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Bishop]
                     + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Rook]
                     + 2 * 1 * evalParams.phaseMaterialValues[(int)Piece::Queen]
                     + 2 * 1 * evalParams.phaseMaterialValues[(int)Piece::King];
    */
    ParamGradient<true> gradient = zeroGradient<true>();
    gradient[getParamIndex(params, params.phaseMaterialValues[(int)Piece::Pawn])] += 2 * 8;
    gradient[getParamIndex(params, params.phaseMaterialValues[(int)Piece::Knight])] += 2 * 2;
    gradient[getParamIndex(params, params.phaseMaterialValues[(int)Piece::Bishop])] += 2 * 2;
    gradient[getParamIndex(params, params.phaseMaterialValues[(int)Piece::Rook])] += 2 * 2;
    gradient[getParamIndex(params, params.phaseMaterialValues[(int)Piece::Queen])] += 2 * 1;
    gradient[getParamIndex(params, params.phaseMaterialValues[(int)Piece::King])] += 2 * 1;
    return gradient;
}

[[nodiscard]] FORCE_INLINE EvalCalcT calcTaperedValue(
        const EvalCalcT earlyValue,
        const EvalCalcT lateValue,
        const EvalCalcT earlyFactor,
        const EvalCalcT lateFactor) {
    return earlyValue * earlyFactor + lateValue * lateFactor;
}

[[nodiscard]] ParamGradient<true> calcTaperedGradient(
        const VectorT& earlyGradient,
        const VectorT& lateGradient,
        const VectorT& earlyFactorGradient,
        const EvalCalcT earlyValue,
        const EvalCalcT lateValue,
        const EvalCalcT earlyFactor,
        const EvalCalcT lateFactor) {
    return earlyGradient * earlyFactor + earlyFactorGradient * earlyValue
         + lateGradient * lateFactor - earlyFactorGradient * lateValue;
}

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE std::tuple<float, float, ParamGradient<CalcJacobians>> calcTaperParams(
        const Evaluator::EvalCalcParams& params,
        const PiecePositionEvaluation& whitePiecePositionEval,
        const PiecePositionEvaluation& blackPiecePositionEval,
        const PiecePositionEvaluationJacobians<CalcJacobians>& whitePiecePositionJacobians,
        const PiecePositionEvaluationJacobians<CalcJacobians>& blackPiecePositionJacobians) {

    const EvalCalcT phaseMaterial =
            whitePiecePositionEval.phaseMaterial + blackPiecePositionEval.phaseMaterial;
    const float earlyFactor = (float)phaseMaterial / (float)params.maxPhaseMaterial;
    const float lateFactor  = 1.f - earlyFactor;

    ParamGradient<CalcJacobians> earlyFactorGradient;
    if constexpr (CalcJacobians) {
        const ParamGradient<CalcJacobians> phaseMaterialGradient =
                whitePiecePositionJacobians.phaseMaterialJacobians
                + blackPiecePositionJacobians.phaseMaterialJacobians;

        const ParamGradient<CalcJacobians> maxPhaseMaterialGradient =
                getMaxPhaseMaterialGradient(params);

        earlyFactorGradient = (phaseMaterialGradient * params.maxPhaseMaterial
                               - phaseMaterial * maxPhaseMaterialGradient)
                            / (params.maxPhaseMaterial * params.maxPhaseMaterial);
    }

    return {earlyFactor, lateFactor, earlyFactorGradient};
}

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE std::tuple<EvalCalcT, ParamGradient<CalcJacobians>> calcMaterialEval(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const PiecePositionEvaluation& whitePiecePositionEval,
        const PiecePositionEvaluation& blackPiecePositionEval,
        const PiecePositionEvaluationJacobians<CalcJacobians>& whitePiecePositionJacobians,
        const PiecePositionEvaluationJacobians<CalcJacobians>& blackPiecePositionJacobians,
        const float earlyFactor,
        const float lateFactor,
        const ParamGradient<CalcJacobians>& earlyFactorGradient) {
    const EvalCalcT earlyMaterial =
            whitePiecePositionEval.material.early - blackPiecePositionEval.material.early;
    const EvalCalcT lateMaterial =
            whitePiecePositionEval.material.late - blackPiecePositionEval.material.late;

    EvalCalcT materialEval = calcTaperedValue(earlyMaterial, lateMaterial, earlyFactor, lateFactor);

    ParamGradient<CalcJacobians> materialGradient;
    if constexpr (CalcJacobians) {
        const ParamGradient<CalcJacobians> earlyMaterialGradient =
                whitePiecePositionJacobians.material.early
                - blackPiecePositionJacobians.material.early;

        const ParamGradient<CalcJacobians> lateMaterialGradient =
                whitePiecePositionJacobians.material.late
                - blackPiecePositionJacobians.material.late;

        materialGradient = calcTaperedGradient(
                earlyMaterialGradient,
                lateMaterialGradient,
                earlyFactorGradient,
                earlyMaterial,
                lateMaterial,
                earlyFactor,
                lateFactor);
    }

    return {materialEval, materialGradient};
}

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE std::tuple<EvalCalcT, ParamGradient<CalcJacobians>> calcPositionEval(
        const PiecePositionEvaluation& whitePiecePositionEval,
        const PiecePositionEvaluation& blackPiecePositionEval,
        const PiecePositionEvaluationJacobians<CalcJacobians>& whitePiecePositionJacobians,
        const PiecePositionEvaluationJacobians<CalcJacobians>& blackPiecePositionJacobians,
        const float earlyFactor,
        const float lateFactor,
        const ParamGradient<CalcJacobians>& earlyFactorGradient) {
    const EvalCalcT earlyPositionEval =
            whitePiecePositionEval.position.early - blackPiecePositionEval.position.early;
    const EvalCalcT latePositionEval =
            whitePiecePositionEval.position.late - blackPiecePositionEval.position.late;
    const EvalCalcT positionEval =
            (EvalCalcT)(earlyPositionEval * earlyFactor + latePositionEval * lateFactor);

    ParamGradient<CalcJacobians> positionGradient;
    if constexpr (CalcJacobians) {
        const ParamGradient<CalcJacobians> earlyPositionGradient =
                whitePiecePositionJacobians.position.early
                - blackPiecePositionJacobians.position.early;

        const ParamGradient<CalcJacobians> latePositionGradient =
                whitePiecePositionJacobians.position.late
                - blackPiecePositionJacobians.position.late;

        positionGradient = calcTaperedGradient(
                earlyPositionGradient,
                latePositionGradient,
                earlyFactorGradient,
                earlyPositionEval,
                latePositionEval,
                earlyFactor,
                lateFactor);
    }

    return {positionEval, positionGradient};
}

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE EvalCalcT evaluateForWhite(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        ParamGradient<CalcJacobians>& whiteEvalGradient) {

    const BoardPosition whiteKingPosition =
            getFirstSetPosition(gameState.getPieceBitBoard(Side::White, Piece::King));
    const BoardPosition blackKingPosition =
            getFirstSetPosition(gameState.getPieceBitBoard(Side::Black, Piece::King));

    PiecePositionEvaluation whitePiecePositionEval{};
    PiecePositionEvaluation blackPiecePositionEval{};

    PiecePositionEvaluationJacobians<CalcJacobians> whitePiecePositionJacobians;
    PiecePositionEvaluationJacobians<CalcJacobians> blackPiecePositionJacobians;

    const BitBoard whiteKingArea =
            getPieceControlledSquares(Piece::King, whiteKingPosition, BitBoard::Empty);
    const BitBoard blackKingArea =
            getPieceControlledSquares(Piece::King, blackKingPosition, BitBoard::Empty);

    const BitBoard kingOverlap      = whiteKingArea & blackKingArea;
    const bool kingsAttackEachOther = kingOverlap != BitBoard::Empty;

    whitePiecePositionEval.control          = whiteKingArea;
    whitePiecePositionEval.kingAttack       = kingOverlap;
    whitePiecePositionEval.numKingAttackers = kingsAttackEachOther;

    blackPiecePositionEval.control          = blackKingArea;
    blackPiecePositionEval.kingAttack       = kingOverlap;
    blackPiecePositionEval.numKingAttackers = kingsAttackEachOther;

    // No need to add king attack weight for the king itself: that would be symmetric.

    evaluatePawnsForSide(
            params,
            gameState,
            Side::White,
            whiteKingPosition,
            blackKingPosition,
            blackKingArea,
            whitePiecePositionEval,
            whitePiecePositionJacobians);
    evaluatePawnsForSide(
            params,
            gameState,
            Side::Black,
            blackKingPosition,
            whiteKingPosition,
            whiteKingArea,
            blackPiecePositionEval,
            blackPiecePositionJacobians);

    evaluatePiecePositionsForSide(
            params,
            gameState,
            Side::White,
            whiteKingPosition,
            blackKingPosition,
            blackKingArea,
            whitePiecePositionEval,
            whitePiecePositionJacobians);
    evaluatePiecePositionsForSide(
            params,
            gameState,
            Side::Black,
            blackKingPosition,
            whiteKingPosition,
            whiteKingArea,
            blackPiecePositionEval,
            blackPiecePositionJacobians);

    evaluateAttackDefend(
            params,
            gameState,
            Side::White,
            whitePiecePositionEval.control,
            blackPiecePositionEval.control,
            whitePiecePositionEval,
            whitePiecePositionJacobians);

    evaluateAttackDefend(
            params,
            gameState,
            Side::Black,
            blackPiecePositionEval.control,
            whitePiecePositionEval.control,
            blackPiecePositionEval,
            blackPiecePositionJacobians);

    const EvalCalcT tempoFactor = gameState.getSideToMove() == Side::White ? 1 : -1;
    updateTaperedTerm(
            params,
            params.tempoBonus,
            whitePiecePositionEval.position,
            whitePiecePositionJacobians.position,
            tempoFactor);

    const auto [earlyFactor, lateFactor, earlyFactorGradient] = calcTaperParams(
            params,
            whitePiecePositionEval,
            blackPiecePositionEval,
            whitePiecePositionJacobians,
            blackPiecePositionJacobians);

    const auto [materialEval, materialGradient] = calcMaterialEval(
            params,
            gameState,
            whitePiecePositionEval,
            blackPiecePositionEval,
            whitePiecePositionJacobians,
            blackPiecePositionJacobians,
            earlyFactor,
            lateFactor,
            earlyFactorGradient);

    const auto [positionEval, positionGradient] = calcPositionEval(
            whitePiecePositionEval,
            blackPiecePositionEval,
            whitePiecePositionJacobians,
            blackPiecePositionJacobians,
            earlyFactor,
            lateFactor,
            earlyFactorGradient);

    EvalCalcT whiteEval = materialEval + positionEval;
    if constexpr (CalcJacobians) {
        whiteEvalGradient = materialGradient + positionGradient;
    }

    correctForDrawish<CalcJacobians>(gameState, params, whiteEval, whiteEvalGradient);

    return whiteEval;
}

[[nodiscard]] FORCE_INLINE std::pair<bool, bool> insufficientMaterialForSides(
        const GameState& gameState) {
    bool whiteInsufficientMaterial = false;
    bool blackInsufficientMaterial = false;

    const bool noWhitePawns =
            gameState.getPieceBitBoard(Side::White, Piece::Pawn) == BitBoard::Empty;
    const bool noWhiteRooks =
            gameState.getPieceBitBoard(Side::White, Piece::Rook) == BitBoard::Empty;
    const bool noWhiteQueens =
            gameState.getPieceBitBoard(Side::White, Piece::Queen) == BitBoard::Empty;

    const bool whiteOnlyHasMinorPieces = noWhitePawns && noWhiteRooks && noWhiteQueens;

    const int numWhiteKnights = popCount(gameState.getPieceBitBoard(Side::White, Piece::Knight));
    const int numWhiteBishops = popCount(gameState.getPieceBitBoard(Side::White, Piece::Bishop));

    const bool whiteOnlyHasAKing =
            whiteOnlyHasMinorPieces && numWhiteKnights == 0 && numWhiteBishops == 0;

    const bool noBlackPawns =
            gameState.getPieceBitBoard(Side::Black, Piece::Pawn) == BitBoard::Empty;
    const bool noBlackRooks =
            gameState.getPieceBitBoard(Side::Black, Piece::Rook) == BitBoard::Empty;
    const bool noBlackQueens =
            gameState.getPieceBitBoard(Side::Black, Piece::Queen) == BitBoard::Empty;

    const bool blackOnlyHasMinorPieces = noBlackPawns && noBlackRooks && noBlackQueens;

    const int numBlackKnights = popCount(gameState.getPieceBitBoard(Side::Black, Piece::Knight));
    const int numBlackBishops = popCount(gameState.getPieceBitBoard(Side::Black, Piece::Bishop));

    const bool blackOnlyHasAKing =
            blackOnlyHasMinorPieces && numBlackKnights == 0 && numBlackBishops == 0;

    if (whiteOnlyHasMinorPieces) {
        if (numWhiteKnights == 0 && numWhiteBishops <= 1) {
            whiteInsufficientMaterial = true;
        } else if (numWhiteBishops == 0 && numWhiteKnights <= 1) {
            whiteInsufficientMaterial = true;
        }
    }

    if (blackOnlyHasMinorPieces) {
        if (numBlackKnights == 0 && numBlackBishops <= 1) {
            blackInsufficientMaterial = true;
        } else if (numBlackBishops == 0 && numBlackKnights <= 1) {
            blackInsufficientMaterial = true;
        }
    }

    if (whiteOnlyHasAKing && blackOnlyHasMinorPieces && numBlackBishops == 0
        && numBlackKnights == 2) {
        blackInsufficientMaterial = true;
    }

    if (blackOnlyHasAKing && whiteOnlyHasMinorPieces && numWhiteBishops == 0
        && numWhiteKnights == 2) {
        whiteInsufficientMaterial = true;
    }

    return {whiteInsufficientMaterial, blackInsufficientMaterial};
}

}  // namespace

Evaluator::EvalCalcParams::EvalCalcParams(const EvalParams& evalParams) : EvalParams(evalParams) {
    maxPhaseMaterial = 2 * 8 * evalParams.phaseMaterialValues[(int)Piece::Pawn]
                     + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Knight]
                     + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Bishop]
                     + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Rook]
                     + 2 * 1 * evalParams.phaseMaterialValues[(int)Piece::Queen]
                     + 2 * 1 * evalParams.phaseMaterialValues[(int)Piece::King];

    pieceSquareTables = {
            evalParams.pieceSquareTablesWhite,
            getReflectedPieceSquareTables(evalParams.pieceSquareTablesWhite)};
}

Evaluator::Evaluator() : Evaluator(EvalParams::getDefaultParams()) {}

Evaluator::Evaluator(const EvalParams& params) : params_(params) {}

FORCE_INLINE int Evaluator::getPieceSquareValue(
        const Piece piece, BoardPosition position, const Side side) const {
    return (int)params_.pieceSquareTables[(int)side][(int)piece][(int)position].early;
}

EvalCalcT Evaluator::evaluateRaw(const GameState& gameState) const {
    ParamGradient<false> gradient;
    const EvalCalcT rawEvalWhite = evaluateForWhite<false>(params_, gameState, gradient);

    return gameState.getSideToMove() == Side::White ? rawEvalWhite : -rawEvalWhite;
}

EvalWithGradient Evaluator::evaluateWithGradient(const GameState& gameState) const {
    ParamGradient<true> gradient = zeroGradient<true>();
    const EvalCalcT rawEvalWhite = evaluateForWhite<true>(params_, gameState, gradient);

    const EvalCalcT colorFactor = gameState.getSideToMove() == Side::White ? 1 : -1;

    return {.eval = colorFactor * rawEvalWhite, .gradient = colorFactor * gradient};
}

EvalT Evaluator::evaluate(const GameState& gameState) const {
    ParamGradient<false> gradient;
    const EvalCalcT rawEvalWhite = evaluateForWhite<false>(params_, gameState, gradient);

    const EvalT clampedEvalWhite =
            (EvalT)clamp((int)rawEvalWhite, -kMateEval + 1'000, kMateEval - 1'000);

    return gameState.getSideToMove() == Side::White ? clampedEvalWhite : -clampedEvalWhite;
}

FORCE_INLINE int getStaticPieceValue(const Piece piece) {
    static constexpr std::array<int, kNumPieceTypes> kStaticPieceValues = {
            100,     // Pawn
            305,     // Knight
            308,     // Bishop
            563,     // Rook
            950,     // Queen
            20'000,  // King
    };

    return kStaticPieceValues[(int)piece];
}

FORCE_INLINE bool isInsufficientMaterial(const GameState& gameState) {
    const auto [whiteInsufficientMaterial, blackInsufficientMaterial] =
            insufficientMaterialForSides(gameState);

    return whiteInsufficientMaterial && blackInsufficientMaterial;
}

FORCE_INLINE EvalT evaluateNoLegalMoves(const GameState& gameState) {
    if (gameState.isInCheck()) {
        // We're in check and there are no legal moves so we're in checkmate.
        return -kMateEval;
    }

    // We're not in check and there are no legal moves so this is a stalemate.
    return 0;
}
