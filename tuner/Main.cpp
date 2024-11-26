#include "LoadPositions.h"
#include "Optimization.h"
#include "PostProcessing.h"
#include "PreProcessing.h"
#include "ScoredPosition.h"
#include "Utilities.h"

#include "chess-engine-lib/Eval.h"
#include "chess-engine-lib/GameState.h"
#include "chess-engine-lib/Math.h"
#include "chess-engine-lib/MoveOrder.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <ranges>
#include <stdexcept>

#include <cstdlib>

namespace {

std::array<double, kNumEvalParams> getInitialParams() {
    const EvalParams defaultParams = EvalParams::getDefaultParams();
    std::println("Initial params:\n{}\n\n", evalParamsToString(defaultParams));

    return evalParamsToDoubles(defaultParams);
}

std::string getParamsString(const std::array<double, kNumEvalParams>& paramsDouble) {
    return paramsDouble | std::ranges::views::transform([](double d) { return std::to_string(d); })
         | std::ranges::views::join_with(std::string(", ")) | std::ranges::to<std::string>();
}

std::array<double, kNumPieceTypes - 1> getAveragePieceValues(
        const EvalParams& params, const std::vector<ScoredPosition>& positions) {
    std::array<double, kNumPieceTypes - 1> sumPieceValues{};
    std::array<double, kNumPieceTypes - 1> numPieceOccurrences{};

    const Evaluator evaluator(params);

    for (const auto& [gameState, _] : positions) {
        const EvalCalcT referenceEval = evaluator.evaluateRaw(gameState);

        for (int sideIdx = 0; sideIdx < kNumSides; ++sideIdx) {
            const Side side         = (Side)sideIdx;
            const double sideFactor = side == gameState.getSideToMove() ? 1.0 : -1.0;

            for (int pieceIdx = 0; pieceIdx < kNumPieceTypes - 1; ++pieceIdx) {
                BitBoard pieceBitBoard = gameState.getPieceBitBoard(side, (Piece)pieceIdx);

                while (pieceBitBoard != BitBoard::Empty) {
                    const BoardPosition position = popFirstSetPosition(pieceBitBoard);

                    GameState copyState = gameState;
                    copyState.removePiece(position);

                    const EvalCalcT eval    = evaluator.evaluateRaw(copyState);
                    const double pieceValue = sideFactor * (referenceEval - eval);

                    sumPieceValues[pieceIdx] += pieceValue;
                    numPieceOccurrences[pieceIdx] += 1.0;
                }
            }
        }
    }

    for (int i = 0; i < kNumPieceTypes - 1; ++i) {
        sumPieceValues[i] /= numPieceOccurrences[i];
    }

    return sumPieceValues;
}

void printAveragePieceValues(const std::array<double, kNumPieceTypes - 1>& averagePieceValues) {
    std::println("Average piece values:");
    for (int i = 0; i < kNumPieceTypes - 1; ++i) {
        std::println("\t{}: {}", pieceToString((Piece)i), averagePieceValues[i]);
    }
}

void printResults(
        const std::array<double, kNumEvalParams>& paramsDouble,
        const std::vector<ScoredPosition>& scoredPositions) {
    const EvalParams params = evalParamsFromDoubles(paramsDouble);
    std::println("Optimized params:\n{}\n", evalParamsToString(params));

    const auto averagePieceValues = getAveragePieceValues(params, scoredPositions);
    printAveragePieceValues(averagePieceValues);

    std::println("\nOptimized param values: {}", getParamsString(paramsDouble));
}

void saveResults(
        const std::array<double, kNumEvalParams> paramsDouble, const std::filesystem::path& path) {
    std::ofstream out(path);
    out << getParamsString(paramsDouble);
}

std::vector<std::pair<std::filesystem::path, int>> parseArgs(int argc, char** argv) {
    if (argc < 3) {
        std::println(
                "Usage: {} <dataPath1> <dropoutRate1> [<dataPath2> <dropoutRate2> ...]", argv[0]);
        std::exit(1);
    }

    try {
        std::vector<std::pair<std::filesystem::path, int>> args;
        for (int i = 1; i + 1 < argc;) {
            const std::filesystem::path dataPath = argv[i++];

            if (!std::filesystem::exists(dataPath)) {
                throw std::invalid_argument(
                        std::format("Path '{}' does not exist", dataPath.string()));
            }

            const int dropoutRate = std::stoi(argv[i++]);

            args.emplace_back(dataPath, dropoutRate);
        }

        return args;
    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) try {
    std::srand(42);

    const auto pathsAndDropOutRates = parseArgs(argc, argv);
    //std::vector<std::pair<std::filesystem::path, int>> pathsAndDropOutRates = {
    //        {R"(D:\annotated-fens\since_virtual_king_mobility_to_tune_old.txt)", 64},
    //        {R"(D:\annotated-fens\first-tune-attempts-vs-untuned.txt)", 16},
    //        {R"(D:\annotated-fens\first-fine-tuning.txt)", 32},
    //        {R"(D:\annotated-fens\more-tapered-eval-terms-tuning.txt)", 16},
    //        {R"(D:\annotated-fens\bad-castling-bonus-tune.txt)", 16},
    //        {R"(D:\annotated-fens\attack-defend.txt)", 16},
    //        {R"(D:\annotated-fens\global-pawn-adjustment.txt)", 8},
    //        {R"(D:\annotated-fens\queen-pawn-adjustment.txt)", 4},
    //        {R"(D:\annotated-fens\tune3-phase.txt)", 24},
    //        {R"(D:\annotated-fens\tune4-phase.txt)", 4},
    //        {R"(D:\annotated-fens\simpler-pawn-adjustment.txt)", 40},
    //        {R"(D:\annotated-fens\bishop-enemy-pawn-color-adjustment.txt)", 4},
    //        {R"(D:\annotated-fens\abs-bishop-pawn-color-adjustment.txt)", 4},
    //        {R"(D:\annotated-fens\pawn-enemy-king-tropism.txt)", 4},
    //        {R"(D:\annotated-fens\full-king-tropism.txt)", 4},
    //        {R"(D:\annotated-fens\full-king-tropism2.txt)", 24},
    //        {R"(D:\annotated-fens\king-special-pawn-tropism.txt)", 4},
    //        {R"(D:\annotated-fens\retune-king-special-pawn-tropism.txt)", 2},
    //        {R"(D:\annotated-fens\control-near-king.txt)", 2},
    //        {R"(D:\annotated-fens\undo-simpler-pawn-adjustment.txt)", 2},
    //        {R"(D:\annotated-fens\tune5-phase.txt)", 2},
    //        {R"(D:\annotated-fens\tuned-drawish-factors.txt)", 1},
    //        {R"(D:\annotated-fens\pin-eval2.txt)", 4},
    //        {R"(D:\annotated-fens\pin-eval3.txt)", 2},
    //        {R"(D:\annotated-fens\king-open-file3.txt)", 1},
    //        {R"(D:\annotated-fens\pst-flipping.txt)", 1},
    //};

    std::array<double, kNumEvalParams> paramsDouble = getInitialParams();

    std::println("Loading positions...");
    std::vector<ScoredPosition> scoredPositions;
    for (const auto pathAndDropoutRate : pathsAndDropOutRates) {
        loadScoredPositions(pathAndDropoutRate.first, pathAndDropoutRate.second, scoredPositions);
    }

    std::println("Quiescing positions...");
    quiescePositions(scoredPositions);

    std::println("Optimizing...");
    optimize(paramsDouble, scoredPositions, /*fixPhaseValues*/ true);

    std::println("Post-processing...");
    postProcess(paramsDouble, scoredPositions);

    printResults(paramsDouble, scoredPositions);

    const std::filesystem::path outputPath = "optimized_params.txt";
    saveResults(paramsDouble, outputPath);
    std::println("Saved optimized params to '{}'", outputPath.string());
} catch (const std::exception& e) {
    std::println("Uncaught exception of type {}: {}", typeid(e).name(), e.what());
    return 1;
}
