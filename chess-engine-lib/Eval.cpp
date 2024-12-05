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

using PstMapping       = std::array<std::int8_t, kSquares>;
using PstMappingBySide = std::array<PstMapping, kNumSides>;

struct PstMappings {
    PstMappingBySide defaultIdx;
    PstMappingBySide whiteSideFlipped;
    PstMappingBySide blackSideFlipped;
    PstMappingBySide bothSidesFlipped;
};

constexpr PstMappings kPstMappings = []() {
    PstMappings mappings{};
    for (int squareIdx = 0; squareIdx < kSquares; ++squareIdx) {
        const BoardPosition whitePosition = (BoardPosition)squareIdx;
        const BoardPosition blackPosition = getVerticalReflection(whitePosition);

        const bool onWhiteSide = rankFromPosition(whitePosition) < 4;

        mappings.defaultIdx[0][squareIdx] = (int)whitePosition;
        mappings.defaultIdx[1][squareIdx] = (int)blackPosition;

        // white side flipped
        {
            const BoardPosition flippedWhitePosition =
                    onWhiteSide ? getHorizontalReflection(whitePosition) : whitePosition;
            const BoardPosition flippedBlackPosition =
                    onWhiteSide ? getHorizontalReflection(blackPosition) : blackPosition;

            mappings.whiteSideFlipped[0][squareIdx] = (int)flippedWhitePosition;
            mappings.whiteSideFlipped[1][squareIdx] = (int)flippedBlackPosition;
        }

        // black side flipped
        {
            const BoardPosition flippedWhitePosition =
                    onWhiteSide ? whitePosition : getHorizontalReflection(whitePosition);
            const BoardPosition flippedBlackPosition =
                    onWhiteSide ? blackPosition : getHorizontalReflection(blackPosition);

            mappings.blackSideFlipped[0][squareIdx] = (int)flippedWhitePosition;
            mappings.blackSideFlipped[1][squareIdx] = (int)flippedBlackPosition;
        }

        // both sides flipped
        {
            const BoardPosition flippedWhitePosition = getHorizontalReflection(whitePosition);
            const BoardPosition flippedBlackPosition = getHorizontalReflection(blackPosition);

            mappings.bothSidesFlipped[0][squareIdx] = (int)flippedWhitePosition;
            mappings.bothSidesFlipped[1][squareIdx] = (int)flippedBlackPosition;
        }
    }

    return mappings;
}();

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
struct TermWithGradient;

template <>
struct TermWithGradient<false> {
    EvalCalcT value = 0;
};

template <>
struct TermWithGradient<true> {
    EvalCalcT value          = 0;
    ParamGradient<true> grad = zeroGradient<true>();
};

template <bool CalcJacobians>
struct TaperedEvaluation {
    TermWithGradient<CalcJacobians> early{};
    TermWithGradient<CalcJacobians> late{};
};

template <bool CalcJacobians>
struct PiecePositionEvaluation {
    TaperedEvaluation<CalcJacobians> eval{};

    TermWithGradient<CalcJacobians> phaseMaterial{};

    BitBoard control     = BitBoard::Empty;
    BitBoard kingAttack  = BitBoard::Empty;
    int numKingAttackers = 0;

    const PstMapping* pstIndex;
    const PstMapping* pstIndexKing;
};

FORCE_INLINE std::size_t getParamIndex(const EvalParams& params, const EvalCalcT& param) {
    return (std::size_t)((std::byte*)&param - (std::byte*)&params) / sizeof(EvalCalcT);
}

template <bool CalcJacobians>
FORCE_INLINE void updateTaperedTerm(
        const Evaluator::EvalCalcParams& params,
        const TaperedTerm& term,
        TaperedEvaluation<CalcJacobians>& eval,
        const EvalCalcT weight) {
    eval.early.value += term.early * weight;
    eval.late.value += term.late * weight;

    if constexpr (CalcJacobians) {
        eval.early.grad[params.getParamIndex(term.early)] += weight;
        eval.late.grad[params.getParamIndex(term.late)] += weight;
    }
}

template <bool CalcJacobians, bool IsKing = false>
FORCE_INLINE void updatePiecePositionEvaluation(
        const Evaluator::EvalCalcParams& params,
        const int pieceIdx,
        const BoardPosition position,
        const Side side,
        PiecePositionEvaluation<CalcJacobians>& result) {
    result.phaseMaterial.value += params.phaseMaterialValues[pieceIdx];
    if constexpr (CalcJacobians) {
        result.phaseMaterial.grad[params.getParamIndex(params.phaseMaterialValues[pieceIdx])] += 1;
    }

    // We manually update value and gradient instead of using updateTaperedTerm to account for piece
    // values being folded into the stored PSTs.

    int pstIndex;
    if constexpr (IsKing) {
        pstIndex = (int)((*result.pstIndexKing)[(int)position]);
    } else {
        pstIndex = (int)((*result.pstIndex)[(int)position]);
    }

    result.eval.early.value += params.pieceSquareTablesWhite[pieceIdx][pstIndex].early;
    result.eval.late.value += params.pieceSquareTablesWhite[pieceIdx][pstIndex].late;

    if constexpr (CalcJacobians) {
        result.eval.early.grad[getParamIndex(
                params, params.pieceSquareTablesWhite[pieceIdx][pstIndex].early)] += 1;
        result.eval.late.grad[getParamIndex(
                params, params.pieceSquareTablesWhite[pieceIdx][pstIndex].late)] += 1;

        result.eval.early.grad[params.getParamIndex(params.pieceValues[pieceIdx].early)] += 1;
        result.eval.late.grad[params.getParamIndex(params.pieceValues[pieceIdx].late)] += 1;
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
        PiecePositionEvaluation<CalcJacobians>& result) {
    const BitBoard control    = getPieceControlledSquares(piece, position, anyPiece);
    const BitBoard kingAttack = control & enemyKingArea;

    result.control |= control;
    result.kingAttack |= kingAttack;

    const bool isAttacker = kingAttack != BitBoard::Empty;

    result.numKingAttackers += isAttacker;
    updateTaperedTerm(params, params.kingAttackWeight[(int)piece], result.eval, isAttacker);

    const int mobility = popCount(control & ~ownOccupancy);
    updateTaperedTerm(params, params.mobilityBonus[(int)piece], result.eval, mobility);
}

template <bool CalcJacobians>
FORCE_INLINE void updateForVirtualKingMobility(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BoardPosition kingPosition,
        TaperedEvaluation<CalcJacobians>& eval) {

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

    updateTaperedTerm(params, params.kingVirtualMobilityPenalty, eval, -virtualKingMobility);
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

static constexpr auto kChebyshevDistances = []() {
    std::array<std::array<std::uint8_t, kSquares>, kSquares> distances{};
    for (int from = 0; from < kSquares; ++from) {
        const auto [fromFile, fromRank] = fileRankFromPosition((BoardPosition)from);
        for (int to = 0; to < kSquares; ++to) {
            const auto [toFile, toRank] = fileRankFromPosition((BoardPosition)to);
            const int distance =
                    max(constexprAbs(fromFile - toFile), constexprAbs(fromRank - toRank));
            distances[from][to] = (std::uint8_t)distance;
        }
    }
    return distances;
}();

[[nodiscard]] FORCE_INLINE EvalCalcT
getTropism(const BoardPosition aPos, const BoardPosition bPos) {
    return (EvalCalcT)kTropisms[(int)aPos][(int)bPos];
}

[[nodiscard]] FORCE_INLINE int getChebyshevDistance(
        const BoardPosition aPos, const BoardPosition bPos) {
    return (int)kChebyshevDistances[(int)aPos][(int)bPos];
}

template <bool CalcJacobians>
FORCE_INLINE void updateForKingTropism(
        const Evaluator::EvalCalcParams& params,
        const BoardPosition ownKingPosition,
        const BoardPosition enemyKingPosition,
        const int pieceIdx,
        const BoardPosition position,
        TaperedEvaluation<CalcJacobians>& eval) {
    const EvalCalcT ownTropism   = getTropism(ownKingPosition, position);
    const EvalCalcT enemyTropism = getTropism(enemyKingPosition, position);

    updateTaperedTerm(params, params.ownKingTropism[pieceIdx], eval, ownTropism);

    updateTaperedTerm(params, params.enemyKingTropism[pieceIdx], eval, enemyTropism);
}

template <bool CalcJacobians>
FORCE_INLINE void updateForKingTropism(
        const Evaluator::EvalCalcParams& params,
        const BoardPosition ownKingPosition,
        const BoardPosition enemyKingPosition,
        const Piece piece,
        const BoardPosition position,
        TaperedEvaluation<CalcJacobians>& eval) {
    updateForKingTropism(params, ownKingPosition, enemyKingPosition, (int)piece, position, eval);
}

template <bool CalcJacobians>
FORCE_INLINE void updateForPins(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BoardPosition ownKingPosition,
        const BitBoard anyPiece,
        TaperedEvaluation<CalcJacobians>& eval) {
    const BitBoard pinBitBoard = gameState.getPinBitBoard(side, ownKingPosition);
    if (pinBitBoard == BitBoard::Empty) {
        return;
    }

    for (int pieceIdx = 0; pieceIdx < kNumPieceTypes - 1; ++pieceIdx) {
        const BitBoard& pieceBitBoard = gameState.getPieceBitBoard(side, (Piece)pieceIdx);
        const int numPinnedPieces     = popCount(pieceBitBoard & pinBitBoard);

        updateTaperedTerm(params, params.piecePinnedAdjustment[pieceIdx], eval, numPinnedPieces);
    }
}

template <bool CalcJacobians>
FORCE_INLINE void updateForKingOpenFiles(
        const Evaluator::EvalCalcParams& params,
        const Side side,
        const BoardPosition kingPosition,
        const BitBoard ownPawns,
        TaperedEvaluation<CalcJacobians>& eval) {
    const BitBoard forwardMask = getPawnForwardMask(kingPosition, side);
    if ((ownPawns & forwardMask) == BitBoard::Empty) {
        updateTaperedTerm(params, params.kingOpenFileAdjustment, eval, 1);
    }

    const BitBoard neighboringPawns = ownPawns & getPawnNeighborFileMask(kingPosition);
    float flankWeight               = 0.f;

    const BoardPosition flank1Position =
            (BoardPosition)(((std::uint8_t)kingPosition - (std::uint8_t)1) & 63);
    const BitBoard flank1Mask = getPawnForwardMask(flank1Position, side);
    flankWeight += (float)((neighboringPawns & flank1Mask) == BitBoard::Empty);

    const BoardPosition flank2Position =
            (BoardPosition)(((std::uint8_t)kingPosition + (std::uint8_t)1) & 63);
    const BitBoard flank2Mask = getPawnForwardMask(flank2Position, side);
    flankWeight += (float)((neighboringPawns & flank2Mask) == BitBoard::Empty);

    updateTaperedTerm(params, params.kingFlankOpenFileAdjustment, eval, flankWeight);
}

template <bool CalcJacobians>
void evaluatePiecePositionsForSide(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BoardPosition ownKingPosition,
        const BoardPosition enemyKingPosition,
        const BitBoard enemyKingArea,
        PiecePositionEvaluation<CalcJacobians>& result) {
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
            updateTaperedTerm(params, params.knightPairBonus, result.eval, 1);
        }

        while (pieceBitBoard != BitBoard::Empty) {
            const BoardPosition position = popFirstSetPosition(pieceBitBoard);

            updatePiecePositionEvaluation<CalcJacobians>(
                    params, (int)Piece::Knight, position, side, result);

            updateForKingTropism(
                    params,
                    ownKingPosition,
                    enemyKingPosition,
                    Piece::Knight,
                    position,
                    result.eval);

            updateTaperedTerm(params, params.knightPawnAdjustment[numOwnPawns], result.eval, 1);

            updateMobilityEvaluation<CalcJacobians>(
                    params, Piece::Knight, position, anyPiece, ownOccupancy, enemyKingArea, result);
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
                    params, (int)Piece::Bishop, position, side, result);

            const int squareColor = getSquareColor(position);

            hasBishopOfColor[squareColor] = true;

            updateTaperedTerm(
                    params,
                    params.bishopPawnSameColorAdjustment[ownPawnsPerSquareColor[squareColor]],
                    result.eval,
                    1);

            updateTaperedTerm(
                    params,
                    params.bishopEnemyPawnSameColorAdjustment
                            [enemyPawnsPerSquareColor[squareColor]],
                    result.eval,
                    1);

            updateForKingTropism(
                    params,
                    ownKingPosition,
                    enemyKingPosition,
                    Piece::Bishop,
                    position,
                    result.eval);

            updateMobilityEvaluation<CalcJacobians>(
                    params, Piece::Bishop, position, anyPiece, ownOccupancy, enemyKingArea, result);
        }

        if (hasBishopOfColor[0] && hasBishopOfColor[1]) {
            updateTaperedTerm(params, params.bishopPairBonus, result.eval, 1);
        }
    }

    // Rooks
    {
        BitBoard pieceBitBoard = gameState.getPieceBitBoard(side, Piece::Rook);

        const int numRooks = popCount(pieceBitBoard);
        if (numRooks >= 2) {
            updateTaperedTerm(params, params.rookPairBonus, result.eval, 1);
        }

        while (pieceBitBoard != BitBoard::Empty) {
            const BoardPosition position = popFirstSetPosition(pieceBitBoard);

            updatePiecePositionEvaluation<CalcJacobians>(
                    params, (int)Piece::Rook, position, side, result);

            const BitBoard fileBitBoard = getFileBitBoard(position);
            const bool blockedByOwnPawn = (ownPawns & fileBitBoard) != BitBoard::Empty;
            const bool blockedByAnyPawn = (anyPawn & fileBitBoard) != BitBoard::Empty;

            if (!blockedByAnyPawn) {
                updateTaperedTerm(params, params.rookOpenFileBonus, result.eval, 1);
            } else if (!blockedByOwnPawn) {
                updateTaperedTerm(params, params.rookSemiOpenFileBonus, result.eval, 1);
            }

            updateForKingTropism(
                    params, ownKingPosition, enemyKingPosition, Piece::Rook, position, result.eval);

            updateTaperedTerm(params, params.rookPawnAdjustment[numOwnPawns], result.eval, 1);

            updateMobilityEvaluation<CalcJacobians>(
                    params, Piece::Rook, position, anyPiece, ownOccupancy, enemyKingArea, result);
        }
    }

    // Queens
    {
        BitBoard pieceBitBoard = gameState.getPieceBitBoard(side, Piece::Queen);

        while (pieceBitBoard != BitBoard::Empty) {
            const BoardPosition position = popFirstSetPosition(pieceBitBoard);

            updatePiecePositionEvaluation<CalcJacobians>(
                    params, (int)Piece::Queen, position, side, result);

            updateForKingTropism(
                    params,
                    ownKingPosition,
                    enemyKingPosition,
                    Piece::Queen,
                    position,
                    result.eval);

            updateTaperedTerm(params, params.queenPawnAdjustment[numOwnPawns], result.eval, 1);

            updateMobilityEvaluation<CalcJacobians>(
                    params, Piece::Queen, position, anyPiece, ownOccupancy, enemyKingArea, result);
        }
    }

    // King
    {
        updatePiecePositionEvaluation<CalcJacobians, /*IsKing*/ true>(
                params, (int)Piece::King, ownKingPosition, side, result);

        // no mobility bonus for king

        updateForVirtualKingMobility<CalcJacobians>(
                params, gameState, side, ownKingPosition, result.eval);

        // Note king attack data between kings was calculated during initialization.

        // King safety. Note: we rely on the king being evaluated last, so that king attack data is complete.
        const int controlNearEnemyKing = popCount(result.kingAttack);

        updateTaperedTerm(
                params, params.controlNearEnemyKing[controlNearEnemyKing], result.eval, 1);

        const int numKingAttackersIdx =
                min(result.numKingAttackers, (int)params.numKingAttackersAdjustment.size() - 1);
        updateTaperedTerm(
                params, params.numKingAttackersAdjustment[numKingAttackersIdx], result.eval, 1);

        updateForPins(params, gameState, side, ownKingPosition, anyPiece, result.eval);

        updateForKingOpenFiles(params, side, ownKingPosition, ownPawns, result.eval);
    }
}

template <bool CalcJacobians>
FORCE_INLINE void evaluateAttackDefend(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BitBoard ownControl,
        const BitBoard enemyControl,
        TaperedEvaluation<CalcJacobians>& eval) {
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
                    eval,
                    numRelevantPieces);
        }
    }
}

template <bool CalcJacobians>
FORCE_INLINE void modifyForFactor(
        const Evaluator::EvalCalcParams& params,
        const EvalCalcT& factor,
        TermWithGradient<CalcJacobians>& whiteEval) {
    if constexpr (CalcJacobians) {
        whiteEval.grad *= factor;

        auto& dEvalDFactor = whiteEval.grad[params.getParamIndex(factor)];
        MY_ASSERT(dEvalDFactor == 0);
        dEvalDFactor = whiteEval.value;
    }

    whiteEval.value *= factor;
}

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE bool correctForOppositeColoredBishops(
        const GameState& gameState,
        const Evaluator::EvalCalcParams& params,
        TermWithGradient<CalcJacobians>& whiteEval) {
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
            params, params.oppositeColoredBishopFactor[pawnDelta], whiteEval);

    return true;
}

template <bool CalcJacobians>
void correctForDrawish(
        const GameState& gameState,
        const Evaluator::EvalCalcParams& params,
        TermWithGradient<CalcJacobians>& whiteEval) {
    if (correctForOppositeColoredBishops<CalcJacobians>(gameState, params, whiteEval)) {
        return;
    }

    const bool whiteIsStronger = whiteEval.value >= 0;

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
        modifyForFactor<CalcJacobians>(params, params.singleMinorFactor, whiteEval);
        return;
    }

    // Only two knights; this is insufficient material once the weaker side has lost their material.
    if (strongSideMajorPieces == 0 && strongSideKnights == 2 && strongSideBishops == 0) {
        modifyForFactor<CalcJacobians>(params, params.twoKnightsFactor, whiteEval);
        return;
    }

    // Rook vs a minor piece is drawish.
    if (strongSideRooks == 1 && strongSideQueens == 0 && strongSideMinorPieces == 0
        && (weakSideMinorPieces == 1 && weakSideMajorPieces == 0)) {
        modifyForFactor<CalcJacobians>(params, params.rookVsMinorFactor, whiteEval);
        return;
    }

    // Rook and minor vs rook is drawish.
    if (strongSideRooks == 1 && strongSideQueens == 0 && strongSideMinorPieces == 1
        && weakSideRooks == 1 && weakSideQueens == 0 && weakSideMinorPieces == 0) {
        modifyForFactor<CalcJacobians>(params, params.rookAndMinorVsRookFactor, whiteEval);
        return;
    }
}

[[nodiscard]] FORCE_INLINE BoardPosition
getPromotionSquare(const BoardPosition pawnPosition, const Side pawnSide) {
    return (BoardPosition)((((int)pawnSide - 1) & 56) + ((int)pawnPosition & 7));
}

template <bool CalcJacobians>
void evaluatePawnsForSide(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const Side side,
        const BoardPosition ownKingPosition,
        const BoardPosition enemyKingPosition,
        const BitBoard enemyKingArea,
        PiecePositionEvaluation<CalcJacobians>& result) {
    const BitBoard ownPawns   = gameState.getPieceBitBoard(side, Piece::Pawn);
    const BitBoard enemyPawns = gameState.getPieceBitBoard(nextSide(side), Piece::Pawn);

    BitBoard pawnBitBoard = ownPawns;

    while (pawnBitBoard != BitBoard::Empty) {
        const BoardPosition position = popFirstSetPosition(pawnBitBoard);

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
        int pstIdx     = (int)Piece::Pawn;

        if (isDoubledPawn) {
            updateTaperedTerm(params, params.doubledPawnPenalty, result.eval, -1);

            tropismIdx = EvalParams::kDoubledPawnTropismIdx;
        } else if (isPassedPawn) {
            pstIdx     = EvalParams::kPassedPawnPstIdx;
            tropismIdx = EvalParams::kPassedPawnTropismIdx;

            const BoardPosition promotionSquare = getPromotionSquare(position, side);
            const int promotionDistance = min(5, getChebyshevDistance(promotionSquare, position));

            const bool isEnemyTurn = side != gameState.getSideToMove();
            const int kingDistance =
                    getChebyshevDistance(promotionSquare, enemyKingPosition) - isEnemyTurn;

            const bool outsideKingSquare = promotionDistance < kingDistance;
            updateTaperedTerm(
                    params, params.passedPawnOutsideKingSquare, result.eval, outsideKingSquare);
        } else if (isIsolated) {
            tropismIdx = EvalParams::kIsolatedPawnTropismIdx;
        }

        if (isIsolated) {
            updateTaperedTerm(params, params.isolatedPawnPenalty, result.eval, -1);
        }

        updatePiecePositionEvaluation<CalcJacobians>(params, pstIdx, position, side, result);

        updateForKingTropism(
                params, ownKingPosition, enemyKingPosition, tropismIdx, position, result.eval);
    }

    const BitBoard pawnControl = getPawnControlledSquares(ownPawns, side);
    result.control |= pawnControl;

    const BitBoard kingAttack = enemyKingArea & pawnControl;
    result.kingAttack |= kingAttack;

    const int numAttackedSquares = popCount(kingAttack);
    result.numKingAttackers += numAttackedSquares;

    updateTaperedTerm(
            params, params.kingAttackWeight[(int)Piece::Pawn], result.eval, numAttackedSquares);
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
    gradient[params.getParamIndex(params.phaseMaterialValues[(int)Piece::Pawn])] += 2 * 8;
    gradient[params.getParamIndex(params.phaseMaterialValues[(int)Piece::Knight])] += 2 * 2;
    gradient[params.getParamIndex(params.phaseMaterialValues[(int)Piece::Bishop])] += 2 * 2;
    gradient[params.getParamIndex(params.phaseMaterialValues[(int)Piece::Rook])] += 2 * 2;
    gradient[params.getParamIndex(params.phaseMaterialValues[(int)Piece::Queen])] += 2 * 1;
    gradient[params.getParamIndex(params.phaseMaterialValues[(int)Piece::King])] += 2 * 1;
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
        const PiecePositionEvaluation<CalcJacobians>& whitePiecePositionEval,
        const PiecePositionEvaluation<CalcJacobians>& blackPiecePositionEval) {
    const EvalCalcT phaseMaterial =
            whitePiecePositionEval.phaseMaterial.value + blackPiecePositionEval.phaseMaterial.value;
    const float earlyFactor = (float)phaseMaterial / (float)params.maxPhaseMaterial_;
    const float lateFactor  = 1.f - earlyFactor;

    ParamGradient<CalcJacobians> earlyFactorGradient;
    if constexpr (CalcJacobians) {
        const ParamGradient<CalcJacobians> phaseMaterialGradient =
                whitePiecePositionEval.phaseMaterial.grad
                + blackPiecePositionEval.phaseMaterial.grad;

        const ParamGradient<CalcJacobians> maxPhaseMaterialGradient =
                getMaxPhaseMaterialGradient(params);

        earlyFactorGradient = (phaseMaterialGradient * params.maxPhaseMaterial_
                               - phaseMaterial * maxPhaseMaterialGradient)
                            / (params.maxPhaseMaterial_ * params.maxPhaseMaterial_);
    }

    return {earlyFactor, lateFactor, earlyFactorGradient};
}

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE TermWithGradient<CalcJacobians> calcTaperedEval(
        const Evaluator::EvalCalcParams& params,
        const GameState& gameState,
        const PiecePositionEvaluation<CalcJacobians>& whitePiecePositionEval,
        const PiecePositionEvaluation<CalcJacobians>& blackPiecePositionEval,
        const float earlyFactor,
        const float lateFactor,
        const ParamGradient<CalcJacobians>& earlyFactorGradient) {
    const EvalCalcT earlyEval =
            whitePiecePositionEval.eval.early.value - blackPiecePositionEval.eval.early.value;
    const EvalCalcT lateEval =
            whitePiecePositionEval.eval.late.value - blackPiecePositionEval.eval.late.value;

    TermWithGradient<CalcJacobians> result;

    result.value = calcTaperedValue(earlyEval, lateEval, earlyFactor, lateFactor);

    if constexpr (CalcJacobians) {
        const ParamGradient<CalcJacobians> earlyEvalGradient =
                whitePiecePositionEval.eval.early.grad - blackPiecePositionEval.eval.early.grad;

        const ParamGradient<CalcJacobians> lateEvalGradient =
                whitePiecePositionEval.eval.late.grad - blackPiecePositionEval.eval.late.grad;

        result.grad = calcTaperedGradient(
                earlyEvalGradient,
                lateEvalGradient,
                earlyFactorGradient,
                earlyEval,
                lateEval,
                earlyFactor,
                lateFactor);
    }

    return result;
}

template <bool CalcJacobians>
[[nodiscard]] FORCE_INLINE TermWithGradient<CalcJacobians> evaluateForWhite(
        const Evaluator::EvalCalcParams& params, const GameState& gameState) {

    const BoardPosition whiteKingPosition =
            getFirstSetPosition(gameState.getPieceBitBoard(Side::White, Piece::King));
    const BoardPosition blackKingPosition =
            getFirstSetPosition(gameState.getPieceBitBoard(Side::Black, Piece::King));

    PiecePositionEvaluation<CalcJacobians> whitePiecePositionEval{};
    PiecePositionEvaluation<CalcJacobians> blackPiecePositionEval{};

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

    const bool whiteKingIsOnQueenSide = fileFromPosition(whiteKingPosition) < 4;
    const bool blackKingIsOnQueenSide = fileFromPosition(blackKingPosition) < 4;

    if (!whiteKingIsOnQueenSide && !blackKingIsOnQueenSide) {
        // Both on king side: default
        whitePiecePositionEval.pstIndex     = &kPstMappings.defaultIdx[0];
        whitePiecePositionEval.pstIndexKing = &kPstMappings.defaultIdx[0];

        blackPiecePositionEval.pstIndex     = &kPstMappings.defaultIdx[1];
        blackPiecePositionEval.pstIndexKing = &kPstMappings.defaultIdx[1];
    } else if (whiteKingIsOnQueenSide && !blackKingIsOnQueenSide) {
        // White is on queen side: use flipped white mapping for pieces, and fully flipped mapping
        // for black king
        whitePiecePositionEval.pstIndex     = &kPstMappings.whiteSideFlipped[0];
        whitePiecePositionEval.pstIndexKing = &kPstMappings.defaultIdx[0];

        blackPiecePositionEval.pstIndex     = &kPstMappings.whiteSideFlipped[1];
        blackPiecePositionEval.pstIndexKing = &kPstMappings.bothSidesFlipped[1];
    } else if (!whiteKingIsOnQueenSide && blackKingIsOnQueenSide) {
        // Black is on queen side: use flipped black mapping for pieces, and fully flipped mapping
        // for white king
        whitePiecePositionEval.pstIndex     = &kPstMappings.blackSideFlipped[0];
        whitePiecePositionEval.pstIndexKing = &kPstMappings.bothSidesFlipped[0];

        blackPiecePositionEval.pstIndex     = &kPstMappings.blackSideFlipped[1];
        blackPiecePositionEval.pstIndexKing = &kPstMappings.defaultIdx[1];
    } else {  // whiteKingIsOnQueenSide && blackKingIsOnQueenSide
        // Both on queen side: use fully flipped mapping for both sides
        whitePiecePositionEval.pstIndex     = &kPstMappings.bothSidesFlipped[0];
        whitePiecePositionEval.pstIndexKing = &kPstMappings.bothSidesFlipped[0];

        blackPiecePositionEval.pstIndex     = &kPstMappings.bothSidesFlipped[1];
        blackPiecePositionEval.pstIndexKing = &kPstMappings.bothSidesFlipped[1];
    }

    // No need to add king attack weight for the king itself: that would be symmetric.

    evaluatePawnsForSide(
            params,
            gameState,
            Side::White,
            whiteKingPosition,
            blackKingPosition,
            blackKingArea,
            whitePiecePositionEval);
    evaluatePawnsForSide(
            params,
            gameState,
            Side::Black,
            blackKingPosition,
            whiteKingPosition,
            whiteKingArea,
            blackPiecePositionEval);

    evaluatePiecePositionsForSide(
            params,
            gameState,
            Side::White,
            whiteKingPosition,
            blackKingPosition,
            blackKingArea,
            whitePiecePositionEval);
    evaluatePiecePositionsForSide(
            params,
            gameState,
            Side::Black,
            blackKingPosition,
            whiteKingPosition,
            whiteKingArea,
            blackPiecePositionEval);

    evaluateAttackDefend(
            params,
            gameState,
            Side::White,
            whitePiecePositionEval.control,
            blackPiecePositionEval.control,
            whitePiecePositionEval.eval);

    evaluateAttackDefend(
            params,
            gameState,
            Side::Black,
            blackPiecePositionEval.control,
            whitePiecePositionEval.control,
            blackPiecePositionEval.eval);

    const EvalCalcT tempoFactor = gameState.getSideToMove() == Side::White ? 1 : -1;
    updateTaperedTerm(params, params.tempoBonus, whitePiecePositionEval.eval, tempoFactor);

    const auto [earlyFactor, lateFactor, earlyFactorGradient] =
            calcTaperParams(params, whitePiecePositionEval, blackPiecePositionEval);

    auto taperedEval = calcTaperedEval(
            params,
            gameState,
            whitePiecePositionEval,
            blackPiecePositionEval,
            earlyFactor,
            lateFactor,
            earlyFactorGradient);

    correctForDrawish<CalcJacobians>(gameState, params, taperedEval);

    return taperedEval;
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
    maxPhaseMaterial_ = 2 * 8 * evalParams.phaseMaterialValues[(int)Piece::Pawn]
                      + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Knight]
                      + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Bishop]
                      + 2 * 2 * evalParams.phaseMaterialValues[(int)Piece::Rook]
                      + 2 * 1 * evalParams.phaseMaterialValues[(int)Piece::Queen]
                      + 2 * 1 * evalParams.phaseMaterialValues[(int)Piece::King];

    // Fold piece values into PSTs for more efficient calculation.
    // We still use the piece values to subtract this back out in getPieceSquareValue.
    for (int pieceIdx = 0; pieceIdx < kNumPieceTypes; ++pieceIdx) {
        for (int posIdx = 0; posIdx < kSquares; ++posIdx) {
            pieceSquareTablesWhite[pieceIdx][posIdx].early += pieceValues[pieceIdx].early;

            pieceSquareTablesWhite[pieceIdx][posIdx].late += pieceValues[pieceIdx].late;
        }
    }
}

Evaluator::Evaluator() : Evaluator(EvalParams::getDefaultParams()) {}

Evaluator::Evaluator(const EvalParams& params) : params_(params) {}

FORCE_INLINE int Evaluator::getPieceSquareValue(
        const Piece piece, BoardPosition position, const Side side) const {
    // Subtract out the piece values that we folded into the PSTs. That way the returned value is a
    // good representation of how good this square is for the given piece, which is useful for
    // heuristics in the search.
    // We also choose the early values. The values returned by this function are used for history
    // heuristic initialization, so that makes sense. They're also used for move ordering for which
    // the choice is perhaps a bit arbitrary.

    const int squareIndex = kPstMappings.defaultIdx[(int)side][(int)position];

    return (int)params_.pieceSquareTablesWhite[(int)piece][squareIndex].early
         - params_.pieceValues[(int)piece].early;
}

EvalCalcT Evaluator::evaluateRaw(const GameState& gameState) const {
    const auto rawEvalWhite = evaluateForWhite<false>(params_, gameState);

    return gameState.getSideToMove() == Side::White ? rawEvalWhite.value : -rawEvalWhite.value;
}

EvalWithGradient Evaluator::evaluateWithGradient(const GameState& gameState) const {
    const auto rawEvalWhite = evaluateForWhite<true>(params_, gameState);

    const EvalCalcT colorFactor = gameState.getSideToMove() == Side::White ? 1 : -1;

    return {.eval = colorFactor * rawEvalWhite.value, .gradient = colorFactor * rawEvalWhite.grad};
}

EvalT Evaluator::evaluate(const GameState& gameState) const {
    const auto rawEvalWhite = evaluateForWhite<false>(params_, gameState);

    const EvalT clampedEvalWhite =
            (EvalT)clamp((int)rawEvalWhite.value, -kMateEval + 1'000, kMateEval - 1'000);

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
