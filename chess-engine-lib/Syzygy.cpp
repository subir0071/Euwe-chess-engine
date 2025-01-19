#include "Syzygy.h"

#include "Pyrrhic/tbprobe.h"

#include "Macros.h"
#include "MyAssert.h"

namespace {

[[nodiscard]] FORCE_INLINE std::uint64_t getSyzygyOccupancy(
        const GameState& gameState, const Side side) {
    return (std::uint64_t)gameState.getSideOccupancy(side);
}

[[nodiscard]] FORCE_INLINE std::uint64_t getSyzygyOccupancy(
        const GameState& gameState, const Piece piece) {
    return (std::uint64_t)(
            gameState.getPieceBitBoard(Side::White, piece)
            | gameState.getPieceBitBoard(Side::Black, piece));
}

[[nodiscard]] FORCE_INLINE unsigned getSyzygyEnPassantTarget(const GameState& gameState) {
    const BoardPosition enPassantTarget = gameState.getEnPassantTarget();
    if (enPassantTarget == BoardPosition::Invalid) {
        return 0;
    }
    return (unsigned)enPassantTarget;
}

[[nodiscard]] FORCE_INLINE bool getSyzygySide(const GameState& gameState) {
    const Side side = gameState.getSideToMove();
    return side == Side::White;
}

}  // namespace

void initSyzygy(const std::filesystem::path& syzygyDir) {
    tb_init(syzygyDir.string().c_str());
}

void tearDownSyzygy() {
    tb_free();
}

FORCE_INLINE std::optional<EvalT> probeSyzygyWdl(const GameState& gameState) {
    if (gameState.getPlySinceCaptureOrPawn() != 0) {
        return std::nullopt;
    }
    if (gameState.getCastlingRights() != GameState::CastlingRights::None) {
        return std::nullopt;
    }
    if (gameState.getNumPieces() > TB_LARGEST) {
        return std::nullopt;
    }

    unsigned probeResult = tb_probe_wdl(
            getSyzygyOccupancy(gameState, Side::White),
            getSyzygyOccupancy(gameState, Side::Black),
            getSyzygyOccupancy(gameState, Piece::King),
            getSyzygyOccupancy(gameState, Piece::Queen),
            getSyzygyOccupancy(gameState, Piece::Rook),
            getSyzygyOccupancy(gameState, Piece::Bishop),
            getSyzygyOccupancy(gameState, Piece::Knight),
            getSyzygyOccupancy(gameState, Piece::Pawn),
            getSyzygyEnPassantTarget(gameState),
            getSyzygySide(gameState));

    MY_ASSERT_DEBUG(probeResult != TB_RESULT_FAILED);

    switch (probeResult) {
        case TB_WIN:
            return mateIn(200);

        case TB_CURSED_WIN:
        case TB_DRAW:
        case TB_BLESSED_LOSS:
            return (EvalT)0;

        case TB_LOSS:
            return -mateIn(200);

        case TB_RESULT_FAILED:
            return std::nullopt;
    }
    UNREACHABLE;
}
