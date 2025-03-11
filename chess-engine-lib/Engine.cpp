#include "Engine.h"

#include "MoveSearcher.h"
#include "Syzygy.h"

#include <algorithm>

class Engine::Impl {
  public:
    Impl();
    ~Impl();

    TimeManager& getTimeManager();

    void setFrontEnd(IFrontEnd* frontEnd);

    void newGame();

    [[nodiscard]] SearchInfo findMove(
            const GameState& gameState, const std::vector<Move>& searchMoves);

    void interruptSearch();

    [[nodiscard]] int getDefaultTTableSizeInMb() const;

    void setTTableSize(int requestedSizeInMb);

    [[nodiscard]] EvalT evaluate(const GameState& gameState) const;

    void initializeSyzygy(std::string_view syzygyDir);

  private:
    StackOfVectors<Move> moveStack_;
    TimeManager timeManager_;
    Evaluator evaluator_;
    MoveSearcher moveSearcher_;
    IFrontEnd* frontEnd_ = nullptr;

    bool hasSyzygy_ = false;
};

Engine::Impl::Impl()
    : evaluator_(EvalParams::getDefaultParams(), /*usePawnKingEvalHashTable*/ true),
      moveSearcher_(timeManager_, evaluator_) {
    moveStack_.reserve(1'000);
}

Engine::Impl::~Impl() {
    if (hasSyzygy_) {
        tearDownSyzygy();
        hasSyzygy_ = false;
    }
}

TimeManager& Engine::Impl::getTimeManager() {
    return timeManager_;
}

void Engine::Impl::setFrontEnd(IFrontEnd* frontEnd) {
    frontEnd_ = frontEnd;
    timeManager_.setFrontEnd(frontEnd);
    moveSearcher_.setFrontEnd(frontEnd);

    frontEnd_->addOption(FrontEndOption::createString(
            "SyzygyPath", "", [this](const std::string_view v) { initializeSyzygy(v); }));
}

void Engine::Impl::newGame() {
    moveSearcher_.newGame();
}

SearchInfo Engine::Impl::findMove(
        const GameState& gameState, const std::vector<Move>& searchMoves) {
    int maxDepth = MoveSearcher::kMaxDepth;

    auto allLegalMoves = gameState.generateMoves(moveStack_);
    if (allLegalMoves.size() == 1) {
        // Only one legal move. We still search to get a score and a PV.
        // Search to depth 2 to get a guess for the opponent's follow-up move.
        maxDepth = 2;
    }

    for (const auto& move : searchMoves) {
        bool isLegal = false;
        for (const auto& legalMove : allLegalMoves) {
            if (move == legalMove) {
                isLegal = true;
                break;
            }
        }

        if (!isLegal) {
            throw std::invalid_argument(
                    "Requested move to search is not legal: " + move.toExtendedString());
        }
    }

    bool tbHit = false;

    // Find moves that are optimal based on Syzygy tablebases.
    std::vector<Move> syzygyRootMoves;
    if (hasSyzygy_ && canProbeSyzgyRoot(gameState)) {
        syzygyRootMoves = getSyzygyRootMoves(gameState);

        tbHit = !syzygyRootMoves.empty();
    }

    // If searchMoves was provided, filter optimal moves to that.
    std::vector<Move> syzygySearchMoves;
    if (!syzygyRootMoves.empty() && !searchMoves.empty()) {
        for (const auto& syzygyMove : syzygyRootMoves) {
            if (std::ranges::contains(searchMoves, syzygyMove)) {
                syzygySearchMoves.push_back(syzygyMove);
            }
        }
    }

    // Restrict moves to search to the first valid one of:
    // - TB optimal moves that were included in requested moves to search.
    // - All requested moves to search.
    // - All TB optimal moves.
    // - All moves (nullptr).
    const std::vector<Move>* const movesToSearch = !syzygySearchMoves.empty() ? &syzygySearchMoves
                                                 : !searchMoves.empty()       ? &searchMoves
                                                 : !syzygyRootMoves.empty()   ? &syzygyRootMoves
                                                                              : nullptr;

    if (movesToSearch && movesToSearch->size() == 1 && searchMoves.size() != 1) {
        // Only one move to search. We still search to get a score and a PV.
        // Search to depth 2 to get a guess for the opponent's follow-up move.
        // Don't do this if searchMoves.size() == 1, because that means the user explicitly
        // requested this move to be searched.
        maxDepth = 2;
    }

    moveSearcher_.resetSearchStatistics();
    moveSearcher_.prepareForNewSearch(gameState, movesToSearch, tbHit);

    GameState copyState(gameState);

    std::optional<EvalT> evalGuess = std::nullopt;
    SearchInfo searchInfo;

    int depth = 1;

    const auto rootNodeInfo = moveSearcher_.getRootNodeInfo(gameState);
    if (rootNodeInfo) {
        depth     = rootNodeInfo->depth;
        evalGuess = rootNodeInfo->eval;

        maxDepth = max(maxDepth, depth);
    }

    for (; depth <= maxDepth; ++depth) {
        const auto searchResult =
                moveSearcher_.searchForBestMove(copyState, depth, moveStack_, evalGuess);

        evalGuess = searchResult.eval;

        if (searchResult.principalVariation.size() > 0) {
            searchInfo.principalVariation = std::vector<Move>(
                    searchResult.principalVariation.begin(), searchResult.principalVariation.end());
        }

        const auto searchStatistics = moveSearcher_.getSearchStatistics();

        searchInfo.score      = searchResult.eval;
        searchInfo.depth      = depth;
        searchInfo.statistics = searchStatistics;

        if (searchResult.wasInterrupted) {
            if (frontEnd_) {
                frontEnd_->reportPartialSearch(searchInfo);
            }
            searchInfo.depth -= 1;
            break;
        }

        if (frontEnd_) {
            frontEnd_->reportFullSearch(searchInfo);
        }

        if (isMate(searchResult.eval) && getMateDistanceInPly(searchResult.eval) <= depth) {
            break;
        }

        if (timeManager_.shouldStopAfterFullPly(depth)) {
            break;
        }
    }

    if (frontEnd_) {
        frontEnd_->reportSearchStatistics(searchInfo.statistics);
    }

    return searchInfo;
}

void Engine::Impl::interruptSearch() {
    moveSearcher_.interruptSearch();
}

int Engine::Impl::getDefaultTTableSizeInMb() const {
    return moveSearcher_.getDefaultTTableSizeInMb();
}

void Engine::Impl::setTTableSize(const int requestedSizeInMb) {
    moveSearcher_.setTTableSize(requestedSizeInMb);
}

EvalT Engine::Impl::evaluate(const GameState& gameState) const {
    return evaluator_.evaluate(gameState);
}

void Engine::Impl::initializeSyzygy(std::string_view syzygyDir) {
    if (!syzygyPathIsValid(syzygyDir)) {
        throw std::invalid_argument(
                "invalid syzygy path. Must be a ;-separated list of directories.");
    }

    if (hasSyzygy_) {
        tearDownSyzygy();
        hasSyzygy_ = false;
    }

    if (!syzygyDir.empty()) {
        const bool success = initSyzygy(std::string(syzygyDir));

        if (!success) {
            throw std::invalid_argument("Failed to initialize Syzygy tablebases.");
        }

        hasSyzygy_ = true;
    }

    moveSearcher_.setSyzygyEnabled(hasSyzygy_);
}

// Implementation of interface: forward to implementation

Engine::Engine() : impl_(std::make_unique<Engine::Impl>()) {}

Engine::~Engine() = default;

TimeManager& Engine::getTimeManager() {
    return impl_->getTimeManager();
}

void Engine::setFrontEnd(IFrontEnd* frontEnd) {
    impl_->setFrontEnd(frontEnd);
}

void Engine::newGame() {
    impl_->newGame();
}

SearchInfo Engine::findMove(const GameState& gameState, const std::vector<Move>& searchMoves) {
    return impl_->findMove(gameState, searchMoves);
}

void Engine::interruptSearch() {
    impl_->interruptSearch();
}

int Engine::getDefaultTTableSizeInMb() const {
    return impl_->getDefaultTTableSizeInMb();
}

void Engine::setTTableSize(const int requestedSizeInMb) {
    impl_->setTTableSize(requestedSizeInMb);
}

EvalT Engine::evaluate(const GameState& gameState) const {
    return impl_->evaluate(gameState);
}
