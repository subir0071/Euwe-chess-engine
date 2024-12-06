#include "LoadPositions.h"

#include <execution>
#include <fstream>
#include <print>
#include <ranges>
#include <sstream>
#include <syncstream>

#include <cstdlib>

namespace {

std::vector<ScoredPosition> loadScoredPositions(
        const std::filesystem::path& annotatedFensPath,
        const int dropoutRate,
        std::ostream* logOutput) {
    std::ifstream in(annotatedFensPath);

    std::vector<ScoredPosition> scoredPositions;

    std::string inputLine;
    while (std::getline(in, inputLine)) {
        if (inputLine.empty()) {
            continue;
        }

        if ((std::rand() % dropoutRate) != 0) {
            continue;
        }

        std::stringstream lineSStream(inputLine);

        std::string token;
        lineSStream >> token;

        if (token != "score") {
            continue;
        }

        double score;
        lineSStream >> score;

        lineSStream >> token;
        if (token != "fen") {
            continue;
        }

        lineSStream >> std::ws;

        std::string fen;
        std::getline(lineSStream, fen);

        const GameState gameState = GameState::fromFen(fen);

        scoredPositions.push_back({gameState, score});
    }

    if (logOutput) {
        std::osyncstream out(*logOutput);
        std::println(
                out,
                "Read {} scored positions from {}",
                scoredPositions.size(),
                annotatedFensPath.filename().string());
    }

    return scoredPositions;
}

}  // namespace

std::vector<ScoredPosition> loadScoredPositions(
        std::vector<std::pair<std::filesystem::path, int>> pathsAndDropoutRates,
        const int additionalDropOutRate,
        std::ostream* logOutput) {
    std::vector<std::vector<ScoredPosition>> nestedScoredPositions(pathsAndDropoutRates.size());

    std::transform(
            std::execution::par_unseq,
            pathsAndDropoutRates.begin(),
            pathsAndDropoutRates.end(),
            nestedScoredPositions.begin(),
            [&](const auto& pathAndDropoutRate) {
                return loadScoredPositions(
                        pathAndDropoutRate.first,
                        pathAndDropoutRate.second * additionalDropOutRate,
                        logOutput);
            });

    return std::ranges::views::join(nestedScoredPositions)
         | std::ranges::to<std::vector<ScoredPosition>>();
}
