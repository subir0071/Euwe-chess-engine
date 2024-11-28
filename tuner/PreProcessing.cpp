#include "PreProcessing.h"

#include "chess-engine-lib/Eval.h"
#include "chess-engine-lib/Math.h"
#include "chess-engine-lib/MoveOrder.h"

#include <algorithm>
#include <execution>
#include <print>
#include <ranges>

namespace {

bool isDraw(const GameState& gameState, StackOfVectors<Move>& stack) {
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

void updateMateDistance(EvalT& score) {
    if (isMate(score)) {
        score = mateDistancePlus1(score);
    }
}

std::pair<EvalT, GameState> quiesce(
        GameState& gameState,
        EvalT alpha,
        EvalT beta,
        StackOfVectors<Move>& stack,
        StackOfVectors<MoveEvalT>& moveScoreStack,
        const Evaluator& evaluator) {
    constexpr EvalT kDeltaPruningThreshold = 200;

    if (isDraw(gameState, stack)) {
        return {0, gameState};
    }

    const BitBoard enemyControl = gameState.getEnemyControl();
    const bool isInCheck        = gameState.isInCheck(enemyControl);

    EvalT standPat = -kInfiniteEval;
    if (!isInCheck) {
        // Stand pat
        standPat = evaluator.evaluate(gameState);
        if (standPat >= beta) {
            return {standPat, gameState};
        }

        alpha = max(alpha, standPat);
    }

    EvalT bestScore     = standPat;
    GameState bestState = gameState;

    auto moves = gameState.generateMoves(stack, enemyControl, /*capturesOnly =*/!isInCheck);
    if (moves.size() == 0) {
        if (isInCheck) {
            return {-kMateEval, gameState};
        }

        const auto allMoves = gameState.generateMoves(stack, enemyControl);
        if (allMoves.size() == 0) {
            // No legal moves, not in check, so stalemate.
            return {0, gameState};
        }

        return {bestScore, bestState};
    }

    auto moveScores = scoreMovesQuiesce(evaluator, moves, 0, gameState, moveScoreStack);

    for (int moveIdx = 0; moveIdx < moves.size(); ++moveIdx) {
        const Move move = selectBestMove(moves, moveScores, moveIdx);

        const auto unmakeInfo = gameState.makeMove(move);

        auto [score, state] = quiesce(gameState, -beta, -alpha, stack, moveScoreStack, evaluator);
        score               = -score;

        gameState.unmakeMove(move, unmakeInfo);

        updateMateDistance(score);

        alpha = max(alpha, score);
        if (score > bestScore) {
            bestScore = score;
            bestState = state;
        }

        if (alpha >= beta) {
            break;
        }
    }

    return {bestScore, bestState};
}

}  // namespace

void quiescePositions(std::vector<ScoredPosition>& scoredPositions) {
    const Evaluator evaluator(EvalParams::getDefaultParams());

    std::vector<std::optional<ScoredPosition>> maybeQuiescedPositions(scoredPositions.size());

    std::transform(
            std::execution::par_unseq,
            scoredPositions.begin(),
            scoredPositions.end(),
            maybeQuiescedPositions.begin(),
            [&evaluator](ScoredPosition& scoredPosition) -> std::optional<ScoredPosition> {
                const EvalT evalThreshold = 500;

                const EvalT baseEval = evaluator.evaluate(scoredPosition.gameState);
                if (std::abs(baseEval) >= evalThreshold) {
                    return std::nullopt;
                }

                const EvalT deltaThreshold = 50;
                const EvalT alpha          = baseEval - deltaThreshold - 1;
                const EvalT beta           = baseEval + deltaThreshold + 1;

                StackOfVectors<Move> moveStack;
                StackOfVectors<MoveEvalT> moveScoreStack;
                auto [score, state] =
                        quiesce(scoredPosition.gameState,
                                alpha,
                                beta,
                                moveStack,
                                moveScoreStack,
                                evaluator);

                const EvalT evalDelta = std::abs(baseEval - score);
                if (evalDelta >= deltaThreshold || std::abs(score) >= evalThreshold) {
                    return std::nullopt;
                }

                double scoreToUse = scoredPosition.score;
                if (state.getSideToMove() != scoredPosition.gameState.getSideToMove()) {
                    scoreToUse = 1 - scoreToUse;
                }

                // Run move generation so that the pin bit board is pre-calculated, speeding up evaluation.
                (void)state.generateMoves(moveStack);

                return ScoredPosition{state, scoreToUse};
            });

    std::vector<ScoredPosition> quiescedPositions =
            maybeQuiescedPositions | std::views::filter([](const auto& maybePosition) {
                return maybePosition.has_value();
            })
            | std::views::transform([](const auto& maybePosition) { return maybePosition.value(); })
            | std::ranges::to<std::vector<ScoredPosition>>();

    std::println("Obtained {} quiesced positions", quiescedPositions.size());

    std::swap(scoredPositions, quiescedPositions);
}
