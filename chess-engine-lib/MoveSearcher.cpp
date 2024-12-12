#include "MoveSearcher.h"

#include "Eval.h"
#include "Math.h"
#include "MoveOrdering.h"
#include "SEE.h"
#include "TTable.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <limits>
#include <sstream>

#include <cstdint>
#include <cstring>

class MoveSearcher::Impl {
  public:
    Impl(const TimeManager& timeManager, const Evaluator& evaluator);

    void setFrontEnd(IFrontEnd* frontEnd);

    void newGame();

    [[nodiscard]] RootSearchResult searchForBestMove(
            GameState& gameState,
            int depth,
            StackOfVectors<Move>& stack,
            std::optional<EvalT> evalGuess = std::nullopt);

    void prepareForNewSearch(const GameState& gameState);

    void interruptSearch();

    [[nodiscard]] SearchStatistics getSearchStatistics() const;

    void resetSearchStatistics();

    int getDefaultTTableSizeInMb() const;

    void setTTableSize(int requestedSizeInMb);

  private:
    // == Types ==

    // Outcome of searching a single move; signals to main search whether to continue or stop.
    enum class SearchMoveOutcome {
        Continue,
        Cutoff,
        Interrupted,
    };

    // == Helper functions ==

    // Write updated information to the ttable.
    void updateTTable(
            EvalT bestScore,
            EvalT alphaOrig,
            EvalT beta,
            bool stoppedEarly,
            Move bestMove,
            int depth,
            HashT hash);

    // Extract the principal variation from the transposition table.
    [[nodiscard]] StackVector<Move> extractPv(
            GameState gameState, StackOfVectors<Move>& stack, int depth);

    [[nodiscard]] bool shouldStopSearch() const;

    // == Search functions ==

    // Main search function: alpha-beta search with negamax and transposition table.
    //
    // If returned value s satisfies alpha < s < beta, the value is exact.
    // If s <= alpha, the value is an upper bound.
    // If beta <= s, the value is a lower bound.
    // If stopSearch_ is true, returns std::nullopt
    [[nodiscard]] EvalT search(
            GameState& gameState,
            int depth,
            int ply,
            EvalT alpha,
            EvalT beta,
            Move lastMove,
            int lastNullMovePly,
            StackOfVectors<Move>& stack);

    // Quiescence search. When in check search all moves, when not in check only search captures.
    // Continue until no more capture are available or we get a beta cutoff.
    // When not in check use a stand pat evaluation to set alpha and possibly get a beta cutoff.
    [[nodiscard]] EvalT quiesce(
            GameState& gameState, EvalT alpha, EvalT beta, int ply, StackOfVectors<Move>& stack);

    // Subroutine for search.
    // Search a single move, updating alpha, bestScore and bestMove as necessary.
    [[nodiscard]] SearchMoveOutcome searchMove(
            GameState& gameState,
            const Move& move,
            MoveType moveType,
            int depth,
            int reduction,
            int ply,
            EvalT& alpha,
            EvalT beta,
            StackOfVectors<Move>& stack,
            EvalT& bestScore,
            Move& bestMove,
            const Move& lastMove,
            int lastNullMovePly,
            bool useScoutSearch);

    // Perform an aspiration window search.
    [[nodiscard]] RootSearchResult aspirationWindowSearch(
            GameState& gameState,
            const int depth,
            StackOfVectors<Move>& stack,
            const EvalT initialGuess);

    // == Data ==

    std::atomic<bool> stopSearch_ = false;
    mutable bool wasInterrupted_  = false;

    std::uint8_t tTableTick_ = 0;

    SearchTTable tTable_ = {};

    MoveScorer moveScorer_;

    SearchStatistics searchStatistics_ = {};

    const IFrontEnd* frontEnd_ = nullptr;

    const TimeManager& timeManager_;

    const Evaluator& evaluator_;
};

namespace {

[[nodiscard]] FORCE_INLINE bool nullMovePruningAllowed(
        const GameState& gameState,
        const bool isPvNode,
        const EvalT beta,
        const bool isInCheck,
        const int depth,
        const int ply,
        const int lastNullMovePly) {
    const bool basicConditions = !isPvNode && !isInCheck && !isMate(beta) && ply > 0 && depth >= 3
                              && ply != lastNullMovePly + 2;
    if (!basicConditions) {
        return false;
    }

    // Allow null move only if the side to move has any (non-pawn) pieces.
    // Having pieces makes zugzwang less likely.
    const Side sideToMove         = gameState.getSideToMove();
    const BitBoard piecesBitBoard = gameState.getPieceBitBoard(sideToMove, Piece::Knight)
                                  | gameState.getPieceBitBoard(sideToMove, Piece::Bishop)
                                  | gameState.getPieceBitBoard(sideToMove, Piece::Rook)
                                  | gameState.getPieceBitBoard(sideToMove, Piece::Queen);

    return piecesBitBoard != BitBoard::Empty;
}

[[nodiscard]] FORCE_INLINE int getDepthExtension(const bool isInCheck, const Move& lastMove) {
    // Check extension
    if (isInCheck) {
        return 1;
    }

    // 7th rank extension
    if (lastMove.pieceToMove == Piece::Pawn) {
        const int toRank = rankFromPosition(lastMove.to);
        if (toRank == 1 || toRank == 6) {
            return 1;
        }
    }

    return 0;
}

[[nodiscard]] FORCE_INLINE int getDepthReduction(
        const Move& move,
        const int moveIdx,
        const bool moveIsLosing,
        const bool isPvNode,
        const int depth,
        const int extension) {
    // Don't apply reductions if we're extending the current node.
    if (extension > 0) {
        return 0;
    }

    // Don't apply reductions in PV nodes.
    if (isPvNode) {
        return 0;
    }

    // Don't apply reductions too close to the horizon.
    static constexpr int kMinDepthForReduction = 3;
    if (depth < kMinDepthForReduction) {
        return 0;
    }

    if (!moveIsLosing) {
        // Don't apply reductions to tactical moves with positive SEE.
        const Piece promotionPiece = getPromotionPiece(move);
        if (isCapture(move) || promotionPiece == Piece::Queen) {
            return 0;
        }
    }

    // Late Move Reduction (LMR)
    static constexpr int kMovesForLmr = 4;
    if (moveIdx >= kMovesForLmr) {
        return 1;
    }

    return 0;
}

FORCE_INLINE void updateMateDistance(EvalT& score) {
    if (isMate(score)) {
        score = mateDistancePlus1(score);
    }
}

// Compute the delta between two wrapping counters
FORCE_INLINE std::int8_t computeWrappingTickDelta(std::uint8_t tickA, std::uint8_t tickB) {
    const std::uint8_t wrappingTickDelta = tickA - tickB;
    std::int8_t signedTickDelta;
    // 2's complement is guaranteed
    // if the absolute value of the true tick difference is <= 127, the resulting signedTickDelta
    // is correct
    std::memcpy(&signedTickDelta, &wrappingTickDelta, sizeof(wrappingTickDelta));
    return signedTickDelta;
}

FORCE_INLINE Move getTTableMove(const SearchTTPayload payload, const GameState& gameState) {
    return Move{
            .pieceToMove = getPiece(gameState.getPieceOnSquare(payload.moveFrom)),
            .from        = payload.moveFrom,
            .to          = payload.moveTo,
            .flags       = payload.moveFlags,
    };
}

[[nodiscard]] FORCE_INLINE bool isDraw(const GameState& gameState, StackOfVectors<Move>& stack) {
    if (gameState.isRepetition(/*repetitionThreshold =*/2)) {
        return true;
    }

    if (gameState.isFiftyMoves()) {
        const auto moves = gameState.generateMoves(stack);
        if (moves.size() == 0) {
            return evaluateNoLegalMoves(gameState);
        } else {
            return true;
        }
    }

    if (isInsufficientMaterial(gameState)) {
        return true;
    }

    return false;
}

}  // namespace

MoveSearcher::Impl::Impl(const TimeManager& timeManager, const Evaluator& evaluator)
    : moveScorer_(evaluator), timeManager_(timeManager), evaluator_(evaluator) {
    setTTableSize(getDefaultTTableSizeInMb());
}

void MoveSearcher::Impl::setFrontEnd(IFrontEnd* frontEnd) {
    frontEnd_ = frontEnd;
}

void MoveSearcher::Impl::newGame() {
    // Reset internal state for the sake of consistency

    stopSearch_     = false;
    wasInterrupted_ = false;

    moveScorer_.newGame();

    tTable_.clear();
    tTableTick_ = 0;

    resetSearchStatistics();
}

FORCE_INLINE void MoveSearcher::Impl::updateTTable(
        const EvalT bestScore,
        const EvalT alphaOrig,
        const EvalT beta,
        const bool stoppedEarly,
        const Move bestMove,
        const int depth,
        const HashT hash) {
    ScoreType scoreType;
    if (stoppedEarly) {
        if (bestScore > alphaOrig) {
            // Alpha was raised (but didn't cause a cut off), and the search was interrupted.
            // So this is a partially searched PV node.
            // The score is a lower bound since we raised alpha. Store the score and the best move so
            // far.
            scoreType = ScoreType::LowerBound;
        } else {
            // No move raised alpha but we didn't search all moves, so we don't know if the score is
            // a lower or upper bound.
            scoreType = ScoreType::NotSet;
        }
    } else if (bestScore <= alphaOrig) {
        // Best score is below original feasibility window, so it's an upper bound.
        scoreType = ScoreType::UpperBound;
    } else if (bestScore >= beta) {
        // A score above beta was obtained from a subcall that failed high, so that result is an
        // upper bound. This is true regardless of whether the score is above the original beta or
        // a tightened beta.
        scoreType = ScoreType::LowerBound;
    } else {
        // Score is in the feasibility window, so exact.
        scoreType = ScoreType::Exact;
    }

    const SearchTTable::EntryT entry = {
            .hash    = hash,
            .payload = {
                    .score     = bestScore,
                    .depth     = (std::uint8_t)depth,
                    .tick      = tTableTick_,
                    .scoreType = scoreType,
                    .moveFrom  = bestMove.from,
                    .moveTo    = bestMove.to,
                    .moveFlags = bestMove.flags,
            }};

    const auto isMoreValuable = [](const SearchTTPayload& newPayload,
                                   const SearchTTPayload& oldPayload) FORCE_INLINE {
        const int tickDelta        = computeWrappingTickDelta(newPayload.tick, oldPayload.tick);
        const int compensatedDepth = (int)newPayload.depth + tickDelta;
        if (compensatedDepth > (int)oldPayload.depth) {
            // If new entry is deeper (biased for recency), consider it more valuable.
            return true;
        } else if (compensatedDepth == (int)oldPayload.depth) {
            // If the old and new entry are of equal depth:
            //  - If the new entry is exact, consider it more valuable (if old one is also exact, we
            //    prefer the new one).
            //  - If neither the old nor the new entry are exact, we prefer the new one.
            return newPayload.scoreType == ScoreType::Exact
                || oldPayload.scoreType != ScoreType::Exact;
        } else {
            // Older entry is deeper, retain it.
            return false;
        }
    };

    tTable_.store(entry, isMoreValuable);
}

StackVector<Move> MoveSearcher::Impl::extractPv(
        GameState gameState, StackOfVectors<Move>& stack, const int depth) {
    // Note: taking copy of gameState
    // TODO: would make+unmake be faster here?

    StackVector<Move> pv = stack.makeStackVector();

    while (pv.size() < depth) {
        const auto ttHit = tTable_.probe(gameState.getBoardHash());

        if (!ttHit) {
            break;
        }

        const Move move = getTTableMove(ttHit->payload, gameState);

        pv.push_back(move);
        (void)gameState.makeMove(move);
    }

    pv.lock();
    return pv;
}

bool MoveSearcher::Impl::shouldStopSearch() const {
    const std::uint64_t numNodes =
            searchStatistics_.normalNodesSearched + searchStatistics_.qNodesSearched;
    wasInterrupted_ =
            wasInterrupted_ || stopSearch_ || timeManager_.shouldInterruptSearch(numNodes);
    return wasInterrupted_;
}

EvalT MoveSearcher::Impl::search(
        GameState& gameState,
        int depth,
        const int ply,
        EvalT alpha,
        EvalT beta,
        Move lastMove,
        const int lastNullMovePly,
        StackOfVectors<Move>& stack) {
    if (shouldStopSearch()) {
        return -kInfiniteEval;
    }

    const bool isPvNode = beta - alpha > 1;

    ++searchStatistics_.normalNodesSearched;

    if (isPvNode) {
        searchStatistics_.selectiveDepth = max(searchStatistics_.selectiveDepth, ply);
    }

    if (depth == 0) {
        return quiesce(gameState, alpha, beta, ply, stack);
    }

    // alphaOrig determines whether the value returned is an upper bound
    const EvalT alphaOrig = alpha;

    if (ply > 0) {
        if (isDraw(gameState, stack)) {
            // Exact value
            return 0;
        }
    }

    const BitBoard enemyControl = gameState.getEnemyControl();
    const bool isInCheck        = gameState.isInCheck(enemyControl);

    const int extension = getDepthExtension(isInCheck, lastMove);
    if (ply > 0) {
        depth += extension;
    }

    const bool boundsAreMate = isMate(alpha) || isMate(beta);

    constexpr int kMaxFutilityPruningDepth = 3;
    const bool futilityPruningEnabled      = depth <= kMaxFutilityPruningDepth && !boundsAreMate;

    constexpr int kMaxReverseFutilityPruningDepth = 5;
    const bool reverseFutilityPruningEnabled =
            !isPvNode && !isInCheck && depth <= kMaxReverseFutilityPruningDepth && !boundsAreMate;

    EvalT staticEval = -kInfiniteEval;
    if (futilityPruningEnabled || reverseFutilityPruningEnabled) {
        staticEval = evaluator_.evaluate(gameState);
    }

    if (reverseFutilityPruningEnabled) {
        static constexpr EvalT futilityMarginPerDepth = 140;
        const int futilityValue                       = staticEval - futilityMarginPerDepth * depth;

        if (futilityValue >= beta) {
            // Return a conservative lower bound (fail-hard).
            return beta;
        }
    }

    constexpr int kNullMoveReduction = 3;
    if (nullMovePruningAllowed(gameState, isPvNode, beta, isInCheck, depth, ply, lastNullMovePly)) {
        const int nullMoveSearchDepth = max(1, depth - kNullMoveReduction - 1);

        const auto unmakeInfo = gameState.makeNullMove();

        EvalT nullMoveScore =
                -search(gameState,
                        nullMoveSearchDepth,
                        ply + 1,
                        -beta,
                        -beta + 1,
                        /*lastMove =*/{},
                        /*lastNullMovePly =*/ply,
                        stack);

        gameState.unmakeNullMove(unmakeInfo);

        updateMateDistance(nullMoveScore);

        if (wasInterrupted_) {
            return -kInfiniteEval;
        }

        if (nullMoveScore >= beta) {
            // TODO: update ttable? We don't have a best move to store, but we can store a lower
            // bound on the score.

            // Null move failed high, don't bother searching other moves.
            // Return a conservative lower bound (fail-hard).
            return beta;
        }
    }

    EvalT bestScore = -kInfiniteEval;
    Move bestMove{};
    int movesSearched = 0;

    std::optional<Move> hashMove = std::nullopt;

    // Probe the transposition table and use the stored score and/or move if we get a hit.
    const auto ttHit = tTable_.probe(gameState.getBoardHash());
    if (ttHit) {
        const auto& ttInfo = ttHit->payload;

        searchStatistics_.tTableHits++;

        if (ttInfo.depth >= depth) {
            if (ttInfo.scoreType == ScoreType::Exact) {
                if (isPvNode) {
                    searchStatistics_.selectiveDepth =
                            max(searchStatistics_.selectiveDepth, ply + ttInfo.depth);
                }
                // Exact value
                return ttInfo.score;
            } else if (ttInfo.scoreType == ScoreType::LowerBound) {
                // Can safely raise the lower bound for our search window, because the true value
                // is guaranteed to be above this bound.
                alpha = max(alpha, ttInfo.score);
            } else if (ttInfo.scoreType == ScoreType::UpperBound) {
                // Can safely lower the upper bound for our search window, because the true value
                // is guaranteed to be below this bound.
                beta = min(beta, ttInfo.score);
            }
            // Else: score type not set (result from interrupted search).

            // Check if we can return based on tighter bounds from the transposition table.
            if (alpha >= beta) {
                // Based on information from the ttable, we now know that the true value is outside
                // of the feasibility window.
                // If alpha was raised by the tt entry this is a lower bound and we want to return
                // that raised alpha (fail-soft: that's the tightest lower bound we have).
                // If beta was lowered by the tt entry this is an upper bound and we want to return
                // that lowered beta (fail-soft: that's the tightest upper bound we have).
                // So either way we return the tt entry score.
                return ttInfo.score;
            }
        }

        hashMove = getTTableMove(ttInfo, gameState);

        // Try hash move first.
        // Do we need a legality check here for hash collisions?
        const auto outcome = searchMove(
                gameState,
                *hashMove,
                MoveType::HashMove,
                depth,
                /*reduction =*/0,
                ply,
                alpha,
                beta,
                stack,
                bestScore,
                bestMove,
                lastMove,
                lastNullMovePly,
                /*useScoutSearch =*/false);

        if (outcome == SearchMoveOutcome::Interrupted) {
            return bestScore;
        }

        ++movesSearched;

        if (outcome == SearchMoveOutcome::Cutoff) {
            // Fail high

            updateTTable(
                    bestScore,
                    alphaOrig,
                    beta,
                    wasInterrupted_,
                    bestMove,
                    depth,
                    gameState.getBoardHash());

            // Score was obtained from a subcall that failed high, so it was a lower bound for
            // that position. It is also a lower bound for the overall position because we're
            // maximizing.
            return bestScore;
        }
    }

    auto moves = gameState.generateMoves(stack, enemyControl);
    if (moves.size() == 0) {
        // Exact value
        return evaluateNoLegalMoves(gameState);
    }

    auto moveOrderer = moveScorer_.scoreMoves(std::move(moves), hashMove, gameState, lastMove, ply);

    while (const auto maybeMove = moveOrderer.getNextBestMove(gameState)) {
        const Move move = *maybeMove;

        const int reduction = getDepthReduction(
                move, movesSearched, moveOrderer.lastMoveWasLosing(), isPvNode, depth, extension);

        // Futility pruning
        static constexpr EvalT futilityMarginPerDepth        = 140;
        static constexpr EvalT futilityMarginPerMoveSearched = 20;
        const EvalT futilityMargin = futilityMarginPerDepth * (depth - reduction)
                                   - futilityMarginPerMoveSearched * movesSearched;
        const EvalT futilityValue = staticEval + max(futilityMargin, (EvalT)0);
        if (futilityPruningEnabled && futilityValue <= alpha && !isCapture(move)) {
            if (!gameState.givesCheck(move)) {
                if (futilityValue > bestScore) {
                    bestScore = futilityValue;
                    bestMove  = move;
                }
                continue;
            }
        }

        const auto outcome = searchMove(
                gameState,
                move,
                moveOrderer.getLastMoveType(),
                depth,
                reduction,
                ply,
                alpha,
                beta,
                stack,
                bestScore,
                bestMove,
                lastMove,
                lastNullMovePly,
                /*useScoutSearch =*/isPvNode && (movesSearched > 0));

        if (outcome != SearchMoveOutcome::Interrupted) {
            ++movesSearched;
        }

        if (outcome != SearchMoveOutcome::Continue) {
            break;
        }
    }

    if (movesSearched > 0) {
        // If we fully evaluated any positions, update the ttable.
        MY_ASSERT(bestMove.pieceToMove != Piece::Invalid);
        updateTTable(
                bestScore,
                alphaOrig,
                beta,
                wasInterrupted_,
                bestMove,
                depth,
                gameState.getBoardHash());
    }

    // If bestScore <= alphaOrig, then all subcalls returned upper bounds and bestScore is the
    // maximum of these upper bounds, so an upper bound on the overall position. This is ok because
    // we're returning a value below the original alpha.
    //
    // Otherwise, alpha >= bestScore > alphaOrig.
    //
    // If alphaOrig < bestScore < alpha, then all subscalls returned upper bounds, but all of these
    // upper bounds were below the lower bound 'promised' by the transposition table. This situation
    // should never occur!! But in practice it may because of 'search instability'.
    //
    // Otherwise, bestScore = alpha.
    //
    // If alpha was never raised after a subcall, then alpha must have been raised from the ttable.
    // In this case all subcalls returned upper bounds, and at least one of those upper bounds was
    // equal to the lower bound 'promised' by the transposition table. In this case the value must
    // be exactly this lower/upper bound.
    //
    // If alpha was raised at least once, but we're not failing high, then bestScore was set from
    // an exact value, and all other subcalls return upper bounds below that. So the value is exact.
    //
    // If we're failing high relative to the ttable beta, but not relative to the original beta,
    // then the ttable beta was incorrect: we found a lower bound greater than the 'promised' upper
    // bound. This should never occur!! But in practice it may because of 'search instability'.
    //
    // If we're failing high relative to the original beta then we found a lower bound outside the
    // feasibility window so we can safely return a lower bound.
    return bestScore;
}

// Quiescence search. When in check search all moves, when not in check only search captures.
// Continue until no more capture are available or we get a beta cutoff.
// When not in check use a stand pat evaluation to set alpha and possibly get a beta cutoff.
EvalT MoveSearcher::Impl::quiesce(
        GameState& gameState, EvalT alpha, EvalT beta, const int ply, StackOfVectors<Move>& stack) {
    constexpr EvalT kDeltaPruningThreshold = 200;

    EvalT bestScore = -kInfiniteEval;
    Move bestMove{};

    if (shouldStopSearch()) {
        return bestScore;
    }

    ++searchStatistics_.qNodesSearched;

    const bool isPvNode = beta - alpha > 1;

    if (isPvNode) {
        searchStatistics_.selectiveDepth = max(searchStatistics_.selectiveDepth, ply);
    }

    if (isDraw(gameState, stack)) {
        return 0;
    }

    const BitBoard enemyControl = gameState.getEnemyControl();
    const bool isInCheck        = gameState.isInCheck(enemyControl);

    bool completedAnySearch = false;
    const EvalT alphaOrig   = alpha;

    EvalT standPat;
    if (!isInCheck) {
        // Stand pat
        standPat  = evaluator_.evaluate(gameState);
        bestScore = standPat;
        if (bestScore >= beta) {
            return bestScore;
        }

        static constexpr int kStandPatDeltaPruningThreshold = 1'000;
        const EvalT deltaPruningScore = standPat + kStandPatDeltaPruningThreshold;
        if (deltaPruningScore < alpha) {
            // Stand pat is so far below alpha that we have no hope of raising it even if we find a
            // good capture. Return the stand pat evaluation plus a large margin.
            return deltaPruningScore;  // TODO: return alpha instead? (also in delta pruning below)
        }

        alpha = max(alpha, bestScore);
    }

    std::optional<Move> hashMove = std::nullopt;

    // Probe the transposition table and use the stored score and/or move if we get a hit.
    const auto ttHit = tTable_.probe(gameState.getBoardHash());
    if (ttHit) {
        const auto& ttInfo = ttHit->payload;

        searchStatistics_.tTableHits++;

        // No need to check depth: in qsearch, depth == 0.

        if (ttInfo.scoreType == ScoreType::Exact) {
            // Exact value
            return ttInfo.score;
        } else if (ttInfo.scoreType == ScoreType::LowerBound) {
            // Can safely raise the lower bound for our search window, because the true value
            // is guaranteed to be above this bound.
            alpha = max(alpha, ttInfo.score);
        } else if (ttInfo.scoreType == ScoreType::UpperBound) {
            // Can safely lower the upper bound for our search window, because the true value
            // is guaranteed to be below this bound.
            beta = min(beta, ttInfo.score);
        }
        // Else: score type not set (result from interrupted search).

        // Check if we can return based on tighter bounds from the transposition table.
        if (alpha >= beta) {
            // Based on information from the ttable, we now know that the true value is outside
            // of the feasibility window.
            // If alpha was raised by the tt entry this is a lower bound and we want to return
            // that raised alpha (fail-soft: that's the tightest lower bound we have).
            // If beta was lowered by the tt entry this is an upper bound and we want to return
            // that lowered beta (fail-soft: that's the tightest upper bound we have).
            // So either way we return the tt entry score.
            return ttInfo.score;
        }

        hashMove = getTTableMove(ttInfo, gameState);

        bool shouldTryHashMove = true;
        if (!isInCheck) {
            if (!isCapture(hashMove->flags)) {
                shouldTryHashMove = false;
            } else {
                // Let deltaPruningScore = standPat + SEE + kDeltaPruningThreshold
                // if deltaPruningScore < alpha, we can prune the move if it doesn't give check
                // So we need to check if SEE >= alpha - standPat - kDeltaPruningThreshold
                const int seeThreshold = alpha - standPat - kDeltaPruningThreshold;

                const int seeBound =
                        staticExchangeEvaluationBound(gameState, *hashMove, seeThreshold);

                if (seeBound < seeThreshold) {
                    // This move looks like it has no hope of raising alpha, so unless it's a check we
                    // can prune it. For some reason still calculating the upper bound for bestScore
                    // helps even for moves that give check...

                    // If our optimistic estimate of the score of this move is above bestScore, raise
                    // bestScore to match. This should mean that an upper bound returned from this
                    // function if we prune moves is still reliable. Note that this is definitely below
                    // alpha.
                    const EvalT deltaPruningScore = standPat + seeBound + kDeltaPruningThreshold;
                    bestScore                     = max(bestScore, deltaPruningScore);

                    if (!gameState.givesCheck(*hashMove)) {
                        shouldTryHashMove = false;
                    }
                }
            }
        }

        if (shouldTryHashMove) {
            const auto unmakeInfo = gameState.makeMove(*hashMove);

            tTable_.prefetch(gameState.getBoardHash());

            EvalT score = -quiesce(gameState, -beta, -alpha, ply + 1, stack);

            gameState.unmakeMove(*hashMove, unmakeInfo);

            if (wasInterrupted_) {
                return bestScore;
            }

            updateMateDistance(score);

            completedAnySearch = true;
            bestScore          = max(bestScore, score);
            bestMove           = *hashMove;

            if (score >= beta) {
                updateTTable(
                        bestScore,
                        alphaOrig,
                        beta,
                        false,
                        bestMove,
                        /*depth =*/0,
                        gameState.getBoardHash());

                return score;
            }

            alpha = max(alpha, score);
        }
    }

    auto moves = gameState.generateMoves(stack, enemyControl, /*capturesOnly =*/!isInCheck);
    if (moves.size() == 0) {
        if (isInCheck) {
            // We ran full move generation, so no legal moves exist, and we're in check, so it's a
            // checkmate.
            return -kMateEval;
        }

        // No captures are available.

        // Check if we're in an end state by generating all moves.
        // Note that this ignores repetitions and 50 move rule.
        const auto allMoves = gameState.generateMoves(stack, enemyControl);
        if (allMoves.size() == 0) {
            // No legal moves, not in check, so stalemate.
            return 0;
        }

        // If we're not in an end state return the stand pat evaluation.
        return bestScore;
    }

    // Ignore the hash move even if we didn't try it, since that would mean we pruned it.
    auto moveOrderer = moveScorer_.scoreMovesQuiescence(std::move(moves), hashMove, gameState);

    while (const auto maybeMove = moveOrderer.getNextBestMoveQuiescence()) {
        const Move move = *maybeMove;

        if (!isInCheck) {
            // Delta pruning

            MY_ASSERT(isCapture(move));

            // Let deltaPruningScore = standPat + SEE + kDeltaPruningThreshold
            // if deltaPruningScore < alpha, we can prune the move if it doesn't give check
            // So we need to check if SEE >= alpha - standPat - kDeltaPruningThreshold
            const int seeThreshold = alpha - standPat - kDeltaPruningThreshold;

            const int seeBound = staticExchangeEvaluationBound(gameState, move, seeThreshold);

            if (seeBound < seeThreshold) {
                // This move looks like it has no hope of raising alpha, so unless it's a check we
                // can prune it. For some reason still calculating the upper bound for bestScore
                // helps even for moves that give check...

                // If our optimistic estimate of the score of this move is above bestScore, raise
                // bestScore to match. This should mean that an upper bound returned from this
                // function if we prune moves is still reliable. Note that this is definitely below
                // alpha.
                const EvalT deltaPruningScore = standPat + seeBound + kDeltaPruningThreshold;
                bestScore                     = max(bestScore, deltaPruningScore);

                if (!gameState.givesCheck(move)) {
                    continue;
                }
            }
        }

        const auto unmakeInfo = gameState.makeMove(move);

        tTable_.prefetch(gameState.getBoardHash());

        EvalT score = -quiesce(gameState, -beta, -alpha, ply + 1, stack);

        gameState.unmakeMove(move, unmakeInfo);

        if (wasInterrupted_) {
            break;
        }

        updateMateDistance(score);

        if (!completedAnySearch) {
            completedAnySearch = true;
            bestMove           = move;
            // bestScore might be > score due to stand pat
        }

        alpha = max(alpha, score);
        if (score > bestScore) {
            bestScore = score;
            bestMove  = move;
        }

        if (alpha >= beta) {
            break;
        }
    }

    if (completedAnySearch) {
        updateTTable(
                bestScore,
                alphaOrig,
                beta,
                wasInterrupted_,
                bestMove,
                /*depth =*/0,
                gameState.getBoardHash());
    }

    return bestScore;
}

FORCE_INLINE MoveSearcher::Impl::SearchMoveOutcome MoveSearcher::Impl::searchMove(
        GameState& gameState,
        const Move& move,
        const MoveType moveType,
        const int depth,
        const int reduction,
        const int ply,
        EvalT& alpha,
        const EvalT beta,
        StackOfVectors<Move>& stack,
        EvalT& bestScore,
        Move& bestMove,
        const Move& lastMove,
        const int lastNullMovePly,
        const bool useScoutSearch) {

    auto unmakeInfo = gameState.makeMove(move);

    const int reducedDepth = max(depth - reduction - 1, 0);
    const int fullDepth    = depth - 1;

    const bool isPvNode              = beta - alpha > 1;
    const bool likelyNullMoveAllowed = nullMovePruningAllowed(
            gameState,
            isPvNode,
            /*beta*/ -alpha,
            /*isInCheck =*/false,
            reducedDepth,
            ply + 1,
            lastNullMovePly);

    HashT hashToPrefetch = gameState.getBoardHash();
    if (likelyNullMoveAllowed) {
        updateHashForSideToMove(hashToPrefetch);
    }
    tTable_.prefetch(hashToPrefetch);

    EvalT score;
    if (useScoutSearch) {
        // Zero window (scout) search
        score = -search(
                gameState, reducedDepth, ply + 1, -alpha - 1, -alpha, move, lastNullMovePly, stack);

        if (reduction > 0 && score > alpha && !wasInterrupted_) {
            // Search again without reduction
            score =
                    -search(gameState,
                            fullDepth,
                            ply + 1,
                            -alpha - 1,
                            -alpha,
                            move,
                            lastNullMovePly,
                            stack);
        }

        if (score > alpha && score < beta && !wasInterrupted_) {
            // If the score is within the window, do a full window search.
            score = -search(
                    gameState, fullDepth, ply + 1, -beta, -alpha, move, lastNullMovePly, stack);
        }
    } else {
        MY_ASSERT(beta == alpha + 1 || reduction == 0);

        score = -search(
                gameState, reducedDepth, ply + 1, -beta, -alpha, move, lastNullMovePly, stack);
    }

    gameState.unmakeMove(move, unmakeInfo);

    if (wasInterrupted_) {
        return SearchMoveOutcome::Interrupted;
    }

    updateMateDistance(score);

    if (score > bestScore) {
        bestScore = score;
        bestMove  = move;

        if (bestScore >= beta) {
            moveScorer_.reportCutoff(
                    move, moveType, lastMove, ply, depth, gameState.getSideToMove());

            // Fail high; score is a lower bound.
            return SearchMoveOutcome::Cutoff;
        }

        // If score is above alpha, it is either exact or a lower bound, so it is safe to raise
        // the lower bound of our feasibility window.
        alpha = max(alpha, bestScore);
    }
    moveScorer_.reportNonCutoff(move, moveType, depth, gameState.getSideToMove());

    return SearchMoveOutcome::Continue;
}

// Perform an aspiration window search.
RootSearchResult MoveSearcher::Impl::aspirationWindowSearch(
        GameState& gameState,
        const int depth,
        StackOfVectors<Move>& stack,
        const EvalT initialGuess) {
    static constexpr EvalT kInitialTolerance      = 25;
    static constexpr int kToleranceIncreaseFactor = 4;

    int lowerTolerance = kInitialTolerance;
    int upperTolerance = kInitialTolerance;

    auto toEval = [](int v) {
        return (EvalT)(clamp(v, (int)-kInfiniteEval, (int)kInfiniteEval));
    };

    EvalT lowerBound = toEval(initialGuess - lowerTolerance);
    EvalT upperBound = toEval(initialGuess + upperTolerance);

    bool everFailedLow = false;

    EvalT lastCompletedEval = -kInfiniteEval;

    do {
        const auto searchEval =
                search(gameState,
                       depth,
                       0,
                       lowerBound,
                       upperBound,
                       /*lastMove =*/{},
                       /*lastNullMovePly =*/INT_MIN,
                       stack);

        const bool noEval = searchEval < -kMateEval;
        if (!noEval) {
            lastCompletedEval = searchEval;

            const bool failedLow = searchEval <= lowerBound;
            everFailedLow |= failedLow;
        }

        if (wasInterrupted_) {
            if (everFailedLow) {
                // Don't trust the best move if we ever failed low and didn't complete the search.
                // TODO: we should probably ask for more search time here.

                if (frontEnd_) {
                    frontEnd_->reportDiscardedPv("partial aspiration search with failed low");
                }

                StackVector<Move> principalVariation = stack.makeStackVector();
                principalVariation.lock();
                return {.principalVariation = std::move(principalVariation),
                        .eval               = lastCompletedEval,
                        .wasInterrupted     = true};
            }

            // Return partial result.
            return {.principalVariation = extractPv(gameState, stack, depth),
                    .eval               = lastCompletedEval,
                    .wasInterrupted     = true};
        }

        // If we weren't interrupted we should have a valid eval.
        MY_ASSERT(!noEval);

        if (lowerBound < searchEval && searchEval < upperBound) {
            // Eval is within the aspiration window; return result.
            return {.principalVariation = extractPv(gameState, stack, depth),
                    .eval               = searchEval,
                    .wasInterrupted     = false};
        }

        const EvalT previousLowerBound = lowerBound;
        const EvalT previousUpperBound = upperBound;

        if (searchEval <= lowerBound) {
            // Failed low
            if (isMate(searchEval) && searchEval < 0) {
                // If we found a mate, fully open up the window to the mating side.
                // This will allow us to find the earliest mate.
                lowerBound = -kInfiniteEval;
            } else {
                // Exponentially grow the tolerance.
                const int oldTolerance = lowerTolerance;
                lowerTolerance *= kToleranceIncreaseFactor;
                // Expand the lower bound based on the increased tolerance or the search result,
                // whichever is lower.
                lowerBound = toEval(min(searchEval - oldTolerance, initialGuess - lowerTolerance));
            }
        } else {
            // Failed high
            if (isMate(searchEval) && searchEval > 0) {
                // If we found a mate, fully open up the window to the mating side.
                // This will allow us to find the earliest mate.
                upperBound = kInfiniteEval;
            } else {
                // Exponentially grow the tolerance.
                const int oldTolerance = upperTolerance;
                upperTolerance *= kToleranceIncreaseFactor;
                // Expand the upper bound based on the increased tolerance or the search result,
                // whichever is higher.
                upperBound = toEval(max(searchEval + oldTolerance, initialGuess + upperTolerance));
            }
        }

        if (frontEnd_) {
            frontEnd_->reportAspirationWindowReSearch(
                    previousLowerBound, previousUpperBound, searchEval, lowerBound, upperBound);
        }
    } while (true);
}

// Entry point: perform search and return the principal variation and evaluation.
RootSearchResult MoveSearcher::Impl::searchForBestMove(
        GameState& gameState,
        const int depth,
        StackOfVectors<Move>& stack,
        std::optional<EvalT> evalGuess) {
    searchStatistics_.selectiveDepth = 0;

    moveScorer_.resetCutoffStatistics();

    const auto reportCutoffStatistics = [this]() {
#ifdef TRACK_CUTOFF_STATISTICS
        std::stringstream ss;
        moveScorer_.printCutoffStatistics(ss);
        frontEnd_->reportDebugString(ss.str());
#endif
    };

    if (evalGuess) {
        auto searchResult = aspirationWindowSearch(gameState, depth, stack, *evalGuess);

        reportCutoffStatistics();

        return searchResult;
    } else {
        const auto searchEval =
                search(gameState,
                       depth,
                       0,
                       -kInfiniteEval,
                       kInfiniteEval,
                       /*lastMove =*/{},
                       /*lastNullMovePly =*/INT_MIN,
                       stack);

        reportCutoffStatistics();

        return {.principalVariation = extractPv(gameState, stack, depth),
                .eval               = searchEval,
                .wasInterrupted     = wasInterrupted_};
    }
}

void MoveSearcher::Impl::prepareForNewSearch(const GameState& gameState) {
    // Set state variables to prepare for search.
    stopSearch_     = false;
    wasInterrupted_ = false;

    moveScorer_.prepareForNewSearch(gameState);

    ++tTableTick_;
}

void MoveSearcher::Impl::interruptSearch() {
    // Set stop flag to interrupt search.
    stopSearch_ = true;
}

SearchStatistics MoveSearcher::Impl::getSearchStatistics() const {

    //for (int side = 0; side < kNumSides; ++side) {
    //    for (int piece = 0; piece < kNumPieceTypes; ++piece) {
    //        for (int square = 0; square < kSquares; ++square) {
    //            const int historyCutOff = gHistoryCutOff[side][piece][square];
    //            const int historyUsed   = gHistoryUsed[side][piece][square];

    //            std::println("{};{};{}: {} / {}", side, piece, square, historyCutOff, historyUsed);
    //        }
    //    }
    //}

    SearchStatistics searchStatistics  = searchStatistics_;
    searchStatistics.ttableUtilization = tTable_.getUtilization();

    return searchStatistics;
}

void MoveSearcher::Impl::resetSearchStatistics() {
    searchStatistics_ = {};
}

int MoveSearcher::Impl::getDefaultTTableSizeInMb() const {
    return 16;
}

void MoveSearcher::Impl::setTTableSize(const int requestedSizeInMb) {
    if (requestedSizeInMb == 0) {
        tTable_ = SearchTTable();
    } else {
        const std::size_t tTableSizeInBytes   = (std::size_t)requestedSizeInMb * 1024 * 1024;
        const std::size_t tTableSizeInEntries = tTableSizeInBytes / sizeof(SearchTTable::EntryT);

        tTable_ = SearchTTable(tTableSizeInEntries);
    }
}

// Implementation of interface: forward to implementation

MoveSearcher::MoveSearcher(const TimeManager& timeManager, const Evaluator& evaluator)
    : impl_(std::make_unique<MoveSearcher::Impl>(timeManager, evaluator)) {}

MoveSearcher::~MoveSearcher() = default;

void MoveSearcher::setFrontEnd(IFrontEnd* frontEnd) {
    impl_->setFrontEnd(frontEnd);
}

void MoveSearcher::newGame() {
    impl_->newGame();
}

RootSearchResult MoveSearcher::searchForBestMove(
        GameState& gameState,
        int depth,
        StackOfVectors<Move>& stack,
        std::optional<EvalT> evalGuess) {
    return impl_->searchForBestMove(gameState, depth, stack, evalGuess);
}

void MoveSearcher::prepareForNewSearch(const GameState& gameState) {
    impl_->prepareForNewSearch(gameState);
}

void MoveSearcher::interruptSearch() {
    impl_->interruptSearch();
}

SearchStatistics MoveSearcher::getSearchStatistics() const {
    return impl_->getSearchStatistics();
}

void MoveSearcher::resetSearchStatistics() {
    impl_->resetSearchStatistics();
}

int MoveSearcher::getDefaultTTableSizeInMb() const {
    return impl_->getDefaultTTableSizeInMb();
}

void MoveSearcher::setTTableSize(const int requestedSizeInMb) {
    impl_->setTTableSize(requestedSizeInMb);
}
