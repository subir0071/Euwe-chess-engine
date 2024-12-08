#include "MoveOrderer.h"

#include "Eval.h"
#include "Macros.h"
#include "MyAssert.h"
#include "SEE.h"

namespace {

// Killer and counter bonuses can potentially both apply.
// Make sure that the combined bonus is less than the capture and promotion bonuses.
constexpr int kCaptureBonus     = 3'000;
constexpr int kPromotionBonus   = 3'000;
constexpr int kKillerMoveBonus  = 1'000;
constexpr int kCounterMoveBonus = 500;

[[nodiscard]] FORCE_INLINE MoveEvalT
scoreCapture(const Evaluator& evaluator, const Move& move, const GameState& gameState) {
    MoveEvalT moveScore = kCaptureBonus;

    Piece capturedPiece;
    BoardPosition captureTarget = move.to;
    if (isEnPassant(move.flags)) {
        capturedPiece = Piece::Pawn;
        captureTarget = gameState.getEnPassantTarget();
    } else {
        capturedPiece = getPiece(gameState.getPieceOnSquare(move.to));
    }

    // Most valuable victim, least valuable aggressor (MVV-LVA), but disfavoring captures
    // with the king.
    // We achieve this by adding the value of the victim, and then subtracting the value of
    // the aggressor, but divided by a large enough factor so that the victim's value
    // dominates for all but the king.
    moveScore += getStaticPieceValue(capturedPiece);
    moveScore -= (getStaticPieceValue(move.pieceToMove) >> 5);

    moveScore += evaluator.getPieceSquareValue(
            capturedPiece, captureTarget, nextSide(gameState.getSideToMove()));

    return moveScore;
}

[[nodiscard]] FORCE_INLINE MoveEvalT
scoreQueenPromotion(const Move& move, const GameState& gameState) {
    MoveEvalT moveScore = kPromotionBonus;

    moveScore += getStaticPieceValue(Piece::Queen);
    moveScore -= getStaticPieceValue(Piece::Pawn);

    return moveScore;
}

}  // namespace

FORCE_INLINE OrderedMoves::OrderedMoves(
        StackVector<Move>&& moves, StackVector<MoveEvalT>&& moveScores, const int firstMoveIdx)
    : state_(State::Init),
      moves_(std::move(moves)),
      moveScores_(std::move(moveScores)),
      currentMoveIdx_(firstMoveIdx),
      firstLosingCaptureIdx_(moves_.size()),
      firstQuietIdx_(moves_.size()) {}

FORCE_INLINE std::optional<Move> OrderedMoves::getNextBestMove(const GameState& gameState) {
    MY_ASSERT(
            0 <= firstLosingCaptureIdx_ && firstLosingCaptureIdx_ <= firstQuietIdx_
            && firstQuietIdx_ <= moves_.size());
    MY_ASSERT(0 <= currentMoveIdx_ && currentMoveIdx_ <= moves_.size());

    switch (state_) {
        case State::Init: {
            partitionTacticalMoves();

            state_ = State::GoodTactical;
            [[fallthrough]];
        }

        case State::GoodTactical: {
            MY_ASSERT(currentMoveIdx_ <= firstLosingCaptureIdx_);

            while (currentMoveIdx_ < firstLosingCaptureIdx_) {
                const int bestMoveIdx =
                        findHighestScoringMove(currentMoveIdx_, firstLosingCaptureIdx_);

                const Move bestMove = moves_[bestMoveIdx];

                if (isCapture(bestMove.flags)) {
                    const bool isNonLosing = staticExchangeEvaluationNonLosing(gameState, bestMove);

                    if (!isNonLosing) {
                        // This move is losing based on SEE. Move it to the losing moves list, and find the next
                        // best move.
                        --firstLosingCaptureIdx_;
                        std::swap(moves_[bestMoveIdx], moves_[firstLosingCaptureIdx_]);
                        std::swap(moveScores_[bestMoveIdx], moveScores_[firstLosingCaptureIdx_]);
                        continue;
                    }
                }

                std::swap(moves_[bestMoveIdx], moves_[currentMoveIdx_]);
                std::swap(moveScores_[bestMoveIdx], moveScores_[currentMoveIdx_]);
                ++currentMoveIdx_;

                return bestMove;
            }

            state_          = State::Quiets;
            currentMoveIdx_ = firstQuietIdx_;
            [[fallthrough]];
        }

        case State::Quiets: {
            MY_ASSERT(firstQuietIdx_ <= currentMoveIdx_ && currentMoveIdx_ <= moves_.size());

            if (currentMoveIdx_ < moves_.size()) {
                const int bestMoveIdx = findHighestScoringMove(currentMoveIdx_, moves_.size());

                const Move bestMove = moves_[bestMoveIdx];

                std::swap(moves_[bestMoveIdx], moves_[currentMoveIdx_]);
                std::swap(moveScores_[bestMoveIdx], moveScores_[currentMoveIdx_]);
                ++currentMoveIdx_;

                return bestMove;
            }

            state_          = State::LosingCaptures;
            currentMoveIdx_ = firstLosingCaptureIdx_;
            [[fallthrough]];
        }

        case State::LosingCaptures: {
            MY_ASSERT(
                    firstLosingCaptureIdx_ <= currentMoveIdx_ && currentMoveIdx_ <= firstQuietIdx_);

            if (currentMoveIdx_ < firstQuietIdx_) {
                // We've exhausted all the non-losing moves. Return the best losing move.
                // Here, 'best' is based on the original move scoring.
                // Losing moves are inserted in reverse order of score, so we need to start at the back.
                const int losingMoveIdx     = currentMoveIdx_ - firstLosingCaptureIdx_;
                const int lastLosingMoveIdx = firstQuietIdx_ - 1;
                const int listIdx           = lastLosingMoveIdx - losingMoveIdx;

                ++currentMoveIdx_;

                MY_ASSERT(listIdx >= firstLosingCaptureIdx_ && listIdx < firstQuietIdx_);
                return moves_[listIdx];
            }

            state_ = State::Done;
            [[fallthrough]];
        }

        case State::Done: {
            MY_ASSERT(currentMoveIdx_ == firstQuietIdx_);

            return std::nullopt;
        }
    }

    UNREACHABLE;
}

FORCE_INLINE std::optional<Move> OrderedMoves::getNextBestMoveQuiescence() {
    if (currentMoveIdx_ == moves_.size()) {
        return std::nullopt;
    }
    MY_ASSERT(0 <= currentMoveIdx_ && currentMoveIdx_ < moves_.size());

    const int bestMoveIdx = findHighestScoringMove(currentMoveIdx_, moves_.size());

    const Move bestMove = moves_[bestMoveIdx];

    std::swap(moves_[bestMoveIdx], moves_[currentMoveIdx_]);
    std::swap(moveScores_[bestMoveIdx], moveScores_[currentMoveIdx_]);
    ++currentMoveIdx_;

    return bestMove;
}

FORCE_INLINE bool OrderedMoves::lastMoveWasLosing() const {
    return state_ == State::LosingCaptures;
}

FORCE_INLINE int OrderedMoves::findHighestScoringMove(int startIdx, int endIdx) const {
    // Select best move based on pre-calculated scores using a simple linear search.
    // If the best move is then swapped to the front, repeated calls of this function end up doing
    // a selection sort.

    int bestMoveIdx         = -1;
    MoveEvalT bestMoveScore = std::numeric_limits<MoveEvalT>::lowest();

    for (int moveIdx = startIdx; moveIdx < endIdx; ++moveIdx) {
        if (moveScores_[moveIdx] > bestMoveScore) {
            bestMoveScore = moveScores_[moveIdx];
            bestMoveIdx   = moveIdx;
        }
    }

    MY_ASSERT(bestMoveIdx != -1);

    return bestMoveIdx;
}

FORCE_INLINE void OrderedMoves::partitionTacticalMoves() {
    MY_ASSERT(state_ == State::Init);

    const auto isTactical = [](const Move& move) {
        return isCapture(move.flags) || getPromotionPiece(move.flags) == Piece::Queen;
    };

    // Partition moves into tactical and quiet moves using Hoare's partitioning scheme.
    int i = currentMoveIdx_ - 1;
    int j = moves_.size();

    while (true) {
        do {
            ++i;
        } while (i < j && isTactical(moves_[i]));

        do {
            --j;
        } while (j >= i && !isTactical(moves_[j]));

        if (i >= j) {
            firstQuietIdx_ = i;
            break;
        }

        std::swap(moves_[i], moves_[j]);
        std::swap(moveScores_[i], moveScores_[j]);
    }

    MY_ASSERT(firstQuietIdx_ <= moves_.size());
#ifndef NDEBUG
    for (int i = currentMoveIdx_; i < firstQuietIdx_; ++i) {
        MY_ASSERT(isTactical(moves_[i]));
    }
    for (int i = firstQuietIdx_; i < moves_.size(); ++i) {
        MY_ASSERT(!isTactical(moves_[i]));
    }
#endif

    firstLosingCaptureIdx_ = firstQuietIdx_;
}

MoveOrderer::MoveOrderer(const Evaluator& evaluator) : evaluator_(evaluator) {
    moveScoreStack_.reserve(1'000);
    newGame();
}

FORCE_INLINE void MoveOrderer::reportMoveSearched(
        const Move& move, const int depth, const Side side) {
    updateHistoryForUse(move, depth, side);
}

FORCE_INLINE void MoveOrderer::reportCutoff(
        const Move& move, const Move& lastMove, const int ply, const int depth, const Side side) {
    storeKillerMove(move, ply);
    storeCounterMove(lastMove, move, side);
    updateHistoryForCutoff(move, depth, side);
}

FORCE_INLINE OrderedMoves MoveOrderer::orderMoves(
        StackVector<Move>&& moves,
        const std::optional<Move>& moveToIgnore,
        const GameState& gameState,
        const Move& lastMove,
        const int ply) const {
    int moveIdx = 0;
    if (moveToIgnore) {
        ignoreMove(*moveToIgnore, moves, moveIdx);
    }

    auto moveScores = scoreMoves(moves, moveIdx, gameState, lastMove, ply);

    return OrderedMoves(std::move(moves), std::move(moveScores), moveIdx);
}

FORCE_INLINE OrderedMoves MoveOrderer::orderMovesQuiescence(
        StackVector<Move>&& moves,
        const std::optional<Move>& moveToIgnore,
        const GameState& gameState) const {
    int moveIdx = 0;
    if (moveToIgnore) {
        ignoreMove(*moveToIgnore, moves, moveIdx);
    }

    auto moveScores = scoreMovesQuiesce(moves, moveIdx, gameState);

    return OrderedMoves(std::move(moves), std::move(moveScores), moveIdx);
}

void MoveOrderer::newGame() {
    moveClockForKillerMoves_ = 0;
    killerMoves_             = {};
    counterMoves_            = {};

    initializeHistoryFromPieceSquare();
}

void MoveOrderer::prepareForNewSearch(const GameState& gameState) {
    shiftKillerMoves(gameState.getHalfMoveClock());
    scaleDownHistory();
}

FORCE_INLINE MoveOrderer::KillerMoves& MoveOrderer::getKillerMoves(const int ply) {
    MY_ASSERT(ply < kMaxDepth);
    return killerMoves_[ply];
}

FORCE_INLINE const MoveOrderer::KillerMoves& MoveOrderer::getKillerMoves(const int ply) const {
    MY_ASSERT(ply < kMaxDepth);
    return killerMoves_[ply];
}

FORCE_INLINE void MoveOrderer::storeKillerMove(const Move& move, const int ply) {
    if (isCapture(move.flags) || isPromotion(move.flags)) {
        // Only store 'quiet' moves as killer moves.
        return;
    }

    auto& plyKillerMoves = getKillerMoves(ply);

    if (move == plyKillerMoves[0]) {
        // Don't store the same move twice.
        return;
    }

    // Shift killer moves down and store the new move at the front.
    plyKillerMoves[1] = plyKillerMoves[0];
    plyKillerMoves[0] = move;
}

FORCE_INLINE Move MoveOrderer::getCounterMove(const Move& move, const Side side) const {
    if (move.pieceToMove == Piece::Invalid) {
        return {};
    }
    return counterMoves_[(int)side][(int)move.pieceToMove][(int)move.to];
}

FORCE_INLINE void MoveOrderer::storeCounterMove(
        const Move& lastMove, const Move& counter, const Side side) {
    if (isCapture(counter.flags) || isPromotion(counter.flags)) {
        // Only store 'quiet' moves as counter moves.
        return;
    }
    if (lastMove.pieceToMove == Piece::Invalid) {
        return;
    }
    counterMoves_[(int)side][(int)lastMove.pieceToMove][(int)lastMove.to] = counter;
}

FORCE_INLINE int MoveOrderer::getHistoryWeight(const int depth) {
    return depth * depth;
}

FORCE_INLINE void MoveOrderer::updateHistoryForCutoff(
        const Move& move, const int depth, const Side side) {
    if (isCapture(move.flags) || isPromotion(move.flags)) {
        // Only update history for 'quiet' moves.
        return;
    }

    const int square = (int)move.to;
    const int piece  = (int)move.pieceToMove;

    historyCutOff_[(int)side][piece][square] += getHistoryWeight(depth);
}

FORCE_INLINE void MoveOrderer::updateHistoryForUse(
        const Move& move, const int depth, const Side side) {
    if (isCapture(move.flags) || isPromotion(move.flags)) {
        // Only update history for 'quiet' moves.
        return;
    }

    const int square = (int)move.to;
    const int piece  = (int)move.pieceToMove;

    historyUsed_[(int)side][piece][square] += getHistoryWeight(depth);
}

void MoveOrderer::shiftKillerMoves(const int halfMoveClock) {
    const int shiftAmount = halfMoveClock - moveClockForKillerMoves_;

    for (int ply = 0; ply < kMaxDepth - shiftAmount; ++ply) {
        killerMoves_[ply] = killerMoves_[(std::size_t)ply + shiftAmount];
    }

    moveClockForKillerMoves_ = halfMoveClock;
}

void MoveOrderer::initializeHistoryFromPieceSquare() {
    static constexpr int kNumScaleBits        = 7;   // 128
    static constexpr int kPieceSquareBiasBits = 16;  // ~65k

    for (int side = 0; side < kNumSides; ++side) {
        for (int piece = 0; piece < kNumPieceTypes; ++piece) {
            for (int square = 0; square < kSquares; ++square) {
                int pieceSquareValue = evaluator_.getPieceSquareValue(
                        (Piece)piece, (BoardPosition)square, (Side)side);
                // Get rid of negative values
                pieceSquareValue = clamp(pieceSquareValue + 50, 0, 1000);

                historyCutOff_[side][piece][square] = pieceSquareValue
                                                   << (kPieceSquareBiasBits - kNumScaleBits);

                historyUsed_[side][piece][square] = 1 << kPieceSquareBiasBits;
            }
        }
    }
}

void MoveOrderer::scaleDownHistory() {
    static constexpr int kScaleDownBits = 4;   // 16
    static constexpr int kTargetBits    = 10;  // 1024
    static constexpr int kCountlTarget  = 32 - kTargetBits;

    for (int side = 0; side < kNumSides; ++side) {
        for (int piece = 0; piece < kNumPieceTypes; ++piece) {
            for (int square = 0; square < kSquares; ++square) {
                unsigned& historyUsed       = historyUsed_[side][piece][square];
                const int historyCountlZero = std::countl_zero(historyUsed);

                const int shiftAmount = clamp(historyCountlZero - kCountlTarget, 0, kScaleDownBits);

                historyUsed >>= shiftAmount;
                historyCutOff_[side][piece][square] >>= shiftAmount;
            }
        }
    }
}

FORCE_INLINE void MoveOrderer::ignoreMove(
        const Move& moveToIgnore, StackVector<Move>& moves, int& moveIdx) const {
    const auto hashMoveIt = std::find(moves.begin(), moves.end(), moveToIgnore);
    MY_ASSERT_DEBUG(hashMoveIt != moves.end());
    if (hashMoveIt != moves.end()) {
        std::swap(*hashMoveIt, moves.front());
        ++moveIdx;
    }
}

StackVector<MoveEvalT> MoveOrderer::scoreMoves(
        const StackVector<Move>& moves,
        const int firstMoveIdx,
        const GameState& gameState,
        const Move& lastMove,
        const int ply) const {
    StackVector<MoveEvalT> scores = moveScoreStack_.makeStackVector();

    const auto& historyCutOffs = historyCutOff_[(int)gameState.getSideToMove()];
    const auto& historyUsed    = historyUsed_[(int)gameState.getSideToMove()];
    const auto& killerMoves    = getKillerMoves(ply);
    const Move counterMove     = getCounterMove(lastMove, gameState.getSideToMove());

    for (int i = 0; i < firstMoveIdx; ++i) {
        scores.push_back(0);
    }

    for (int moveIdx = firstMoveIdx; moveIdx < moves.size(); ++moveIdx) {
        const Move& move = moves[moveIdx];

        MoveEvalT moveScore = 0;

        const int cutOffScore = historyCutOffs[(int)move.pieceToMove][(int)move.to];
        const int usedScore   = historyUsed[(int)move.pieceToMove][(int)move.to];

        moveScore += (cutOffScore << 5) / usedScore;

        if (isCapture(move.flags)) {
            moveScore += scoreCapture(evaluator_, move, gameState);
        }

        // If promoting to a queen is not a good move, promoting to a knight, bishop, or rook is
        // probably even worse. So only give an ordering bonus for promoting to a queen.
        if (getPromotionPiece(move.flags) == Piece::Queen) {
            moveScore += scoreQueenPromotion(move, gameState);
        }

        if (!isCapture(move.flags) && !isPromotion(move.flags)) {
            for (const Move& killerMove : killerMoves) {
                if (move == killerMove) {
                    moveScore += kKillerMoveBonus;
                }
            }

            if (move == counterMove) {
                moveScore += kCounterMoveBonus;
            }
        }

        scores.push_back(moveScore);
    }

    scores.lock();
    return scores;
}

StackVector<MoveEvalT> MoveOrderer::scoreMovesQuiesce(
        const StackVector<Move>& moves, const int firstMoveIdx, const GameState& gameState) const {
    StackVector<MoveEvalT> scores = moveScoreStack_.makeStackVector();

    for (int i = 0; i < firstMoveIdx; ++i) {
        scores.push_back(0);
    }

    for (int moveIdx = firstMoveIdx; moveIdx < moves.size(); ++moveIdx) {
        const Move& move = moves[moveIdx];

        MoveEvalT moveScore = 0;

        moveScore -= evaluator_.getPieceSquareValue(
                move.pieceToMove, move.from, gameState.getSideToMove());
        moveScore += evaluator_.getPieceSquareValue(
                move.pieceToMove, move.to, gameState.getSideToMove());

        if (isCapture(move.flags)) {
            moveScore += scoreCapture(evaluator_, move, gameState);
        }

        // If promoting to a queen is not a good move, promoting to a knight, bishop, or rook is
        // probably even worse. So only give an ordering bonus for promoting to a queen.
        if (getPromotionPiece(move.flags) == Piece::Queen) {
            moveScore += scoreQueenPromotion(move, gameState);
        }

        scores.push_back(moveScore);
    }

    scores.lock();
    return scores;
}
