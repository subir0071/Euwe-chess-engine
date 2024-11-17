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
#include <print>
#include <ranges>
#include <stdexcept>
#include <fstream>

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

void printResults(const std::array<double, kNumEvalParams>& paramsDouble) {
    const EvalParams params = evalParamsFromDoubles(paramsDouble);
    std::println("Optimized params:\n{}\n", evalParamsToString(params));

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

int main(int argc, char** argv) {
    std::srand(42);

    const auto pathsAndDropOutRates = parseArgs(argc, argv);

    std::array<double, kNumEvalParams> paramsDouble = getInitialParams();

    std::println("Loading positions...");
    std::vector<ScoredPosition> scoredPositions;
    for (const auto pathAndDropoutRate : pathsAndDropOutRates) {
        loadScoredPositions(pathAndDropoutRate.first, pathAndDropoutRate.second, scoredPositions);
    }

    std::println("Quiescing positions...");
    quiescePositions(scoredPositions);

    std::println("Optimizing...");
    optimize(paramsDouble, scoredPositions, /*fixPhaseValues*/ false);

    std::println("Post-processing...");
    postProcess(paramsDouble, scoredPositions);

    printResults(paramsDouble);

    const std::filesystem::path outputPath = "optimized_params.txt";
    saveResults(paramsDouble, outputPath);
    std::println("Saved optimized params to '{}'", outputPath.string());
}
