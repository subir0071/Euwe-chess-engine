#pragma once

#include "BoardConstants.h"
#include "Move.h"
#include "StackOfVectors.h"

#include <array>
#include <optional>
#include <utility>

using MoveEvalT = int;

class Evaluator;
class GameState;

class MoveOrderer {
  public:
    MoveOrderer(StackVector<Move>&& moves, StackVector<MoveEvalT>&& moveScores, int firstMoveIdx);

    [[nodiscard]] std::optional<Move> getNextBestMove(const GameState& gameState);
    [[nodiscard]] std::optional<Move> getNextBestMoveQuiescence();

    [[nodiscard]] bool lastMoveWasLosing() const;

  private:
    enum class State {
        Init,
        GoodTactical,
        Quiets,
        LosingCaptures,
        Done,
    };

    void partitionTacticalMoves();
    [[nodiscard]] int findHighestScoringMove(int startIdx, int endIdx) const;

    State state_;

    StackVector<Move> moves_;
    StackVector<MoveEvalT> moveScores_;

    int currentMoveIdx_;
    int firstLosingCaptureIdx_;
    int firstQuietIdx_;
};

class MoveScorer {
  public:
    MoveScorer(const Evaluator& evaluator);

    void reportMoveSearched(const Move& move, int depth, Side side);
    void reportCutoff(const Move& move, const Move& lastMove, int ply, int depth, Side side);

    [[nodiscard]] MoveOrderer scoreMoves(
            StackVector<Move>&& moves,
            const std::optional<Move>& moveToIgnore,
            const GameState& gameState,
            const Move& lastMove,
            int ply) const;

    [[nodiscard]] MoveOrderer scoreMovesQuiescence(
            StackVector<Move>&& moves,
            const std::optional<Move>& moveToIgnore,
            const GameState& gameState) const;

    void newGame();
    void prepareForNewSearch(const GameState& gameState);

  private:
    static constexpr std::size_t kNumKillerMoves = 2;
    static constexpr std::size_t kMaxDepth       = 100;

    using KillerMoves         = std::array<Move, kNumKillerMoves>;
    using KillerMovesPerDepth = std::array<KillerMoves, kMaxDepth>;

    using MovePerSquare       = std::array<Move, kSquares>;
    using CounterMovePerPiece = std::array<MovePerSquare, kNumPieceTypes>;
    using CounterMovePerSide  = std::array<CounterMovePerPiece, kNumSides>;

    using HistoryValueT    = unsigned;
    using HistoryPerSquare = std::array<HistoryValueT, kSquares>;
    using HistoryPerPiece  = std::array<HistoryPerSquare, kNumPieceTypes>;
    using HistoryPerSide   = std::array<HistoryPerPiece, kNumSides>;

    [[nodiscard]] KillerMoves& getKillerMoves(int ply);
    [[nodiscard]] const KillerMoves& getKillerMoves(int ply) const;
    void storeKillerMove(const Move& move, int ply);

    [[nodiscard]] Move getCounterMove(const Move& move, Side side) const;
    void storeCounterMove(const Move& lastMove, const Move& counter, Side side);

    [[nodiscard]] static int getHistoryWeight(int depth);
    void updateHistoryForCutoff(const Move& move, int depth, Side side);
    void updateHistoryForUse(const Move& move, int depth, Side side);

    void shiftKillerMoves(int halfMoveClock);
    void initializeHistoryFromPieceSquare();
    void scaleDownHistory();

    void ignoreMove(const Move& moveToIgnore, StackVector<Move>& moves, int& moveIdx) const;

    [[nodiscard]] StackVector<MoveEvalT> scoreMoves(
            const StackVector<Move>& moves,
            const int firstMoveIdx,
            const GameState& gameState,
            const Move& lastMove,
            int ply) const;

    [[nodiscard]] StackVector<MoveEvalT> scoreMovesQuiesce(
            const StackVector<Move>& moves,
            const int firstMoveIdx,
            const GameState& gameState) const;

    mutable StackOfVectors<MoveEvalT> moveScoreStack_ = {};

    int moveClockForKillerMoves_     = 0;
    KillerMovesPerDepth killerMoves_ = {};

    CounterMovePerSide counterMoves_ = {};

    HistoryPerSide historyCutOff_ = {};
    HistoryPerSide historyUsed_   = {};

    const Evaluator& evaluator_;
};
