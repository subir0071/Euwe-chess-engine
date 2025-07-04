#include "PieceControl.h"

#include "BoardConstants.h"
#include "Intrinsics.h"
#include "Macros.h"

#include <array>
#include <utility>

namespace {

constexpr BitBoard computeKnightControlledSquares(const BoardPosition origin) {
    constexpr std::array kFileRankDsts = {std::pair{1, 2}, std::pair{2, 1}};

    const auto [file, rank]    = fileRankFromPosition(origin);
    BitBoard controlledSquares = BitBoard::Empty;
    for (const auto& [fileDst, rankDst] : kFileRankDsts) {
        for (const auto fileSign : kSigns) {
            for (const auto rankSign : kSigns) {
                const int newFile = file + fileSign * fileDst;
                const int newRank = rank + rankSign * rankDst;

                if (newFile < 0 || newFile > 7 || newRank < 0 || newRank > 7) {
                    continue;
                }
                const BoardPosition newPosition = positionFromFileRank(newFile, newRank);
                controlledSquares |= newPosition;
            }
        }
    }
    return controlledSquares;
}

constexpr std::array<BitBoard, kSquares> kKnightControlledSquares = []() {
    std::array<BitBoard, kSquares> knightControlledSquares = {};
    for (int square = 0; square < kSquares; ++square) {
        knightControlledSquares[square] = computeKnightControlledSquares((BoardPosition)square);
    }
    return knightControlledSquares;
}();

template <int N>
constexpr std::uint64_t shift(const std::uint64_t x) {
    if constexpr (N > 0) {
        return x << N;
    } else {
        return x >> -N;
    }
}

template <int N>
constexpr std::uint64_t fillRayNoMask(std::uint64_t seed, const std::uint64_t blockerMask) {
    std::uint64_t result = seed;

    for (int i = 0; i < 6; ++i) {
        seed = shift<N>(seed) & blockerMask;
        result |= seed;
    }
    result |= shift<N>(seed) & blockerMask;

    return shift<N>(result);
}

template <int N, std::uint64_t directionMask>
constexpr std::uint64_t fillRayWithMask(std::uint64_t seed, std::uint64_t blockerMask) {
    blockerMask &= directionMask;

    std::uint64_t result = seed;

    for (int i = 0; i < 6; ++i) {
        seed = shift<N>(seed) & blockerMask;
        result |= seed;
    }
    result |= shift<N>(seed) & blockerMask;

    return shift<N>(result) & directionMask;
}

// NB: includes the origin square
template <int N, std::uint64_t directionMask>
constexpr std::array<std::uint64_t, kSquares> getFullRays() {
    std::array<std::uint64_t, kSquares> rays = {};

    for (int file = 0; file < kFiles; ++file) {
        for (int rank = 0; rank < kRanks; ++rank) {
            const BoardPosition position    = positionFromFileRank(file, rank);
            const std::uint64_t positionBit = 1ULL << (int)position;
            rays[(int)position] = fillRayWithMask<N, directionMask>(positionBit, kAllMask);
            rays[(int)position] |= positionBit;
        }
    }

    return rays;
}

static constexpr int kNorth = 8;
static constexpr int kEast  = 1;
static constexpr int kSouth = -8;
static constexpr int kWest  = -1;

static constexpr int kNorthEast = kNorth + kEast;
static constexpr int kSouthEast = kSouth + kEast;
static constexpr int kSouthWest = kSouth + kWest;
static constexpr int kNorthWest = kNorth + kWest;

constexpr std::array<std::array<std::array<std::uint64_t, kSquares>, 3>, 3> kFullRays = {
        // file -1 (west)
        std::array<std::array<std::uint64_t, kSquares>, 3>{
                // rank -1 (south west)
                std::array<std::uint64_t, kSquares>{getFullRays<kSouthWest, kNotEastFileMask>()},
                // rank 0 (west)
                std::array<std::uint64_t, kSquares>{getFullRays<kWest, kNotEastFileMask>()},
                // rank 1 (north west)
                std::array<std::uint64_t, kSquares>{getFullRays<kNorthWest, kNotEastFileMask>()}},
        // file 0
        std::array<std::array<std::uint64_t, kSquares>, 3>{
                // rank -1 (south)
                std::array<std::uint64_t, kSquares>{getFullRays<kSouth, kAllMask>()},
                // rank 0 (stationary)
                std::array<std::uint64_t, kSquares>{},
                // rank 1 (north)
                std::array<std::uint64_t, kSquares>{getFullRays<kNorth, kAllMask>()}},
        // file 1 (east)
        std::array<std::array<std::uint64_t, kSquares>, 3>{
                // rank -1 (south east)
                std::array<std::uint64_t, kSquares>{getFullRays<kSouthEast, kNotWestFileMask>()},
                // rank 0 (east)
                std::array<std::uint64_t, kSquares>{getFullRays<kEast, kNotWestFileMask>()},
                // rank 1 (north east)
                std::array<std::uint64_t, kSquares>{getFullRays<kNorthEast, kNotWestFileMask>()}}};

BitBoard computeBishopControlledSquares(const BoardPosition origin, const BitBoard anyPiece) {
    const std::uint64_t originBitBoard    = 1ULL << (int)origin;
    const std::uint64_t notOtherPieceMask = ~((std::uint64_t)anyPiece);

    const std::uint64_t northEastControlledSquares =
            fillRayWithMask<9, kNotWestFileMask>(originBitBoard, notOtherPieceMask);
    const std::uint64_t southEastControlledSquares =
            fillRayWithMask<-7, kNotWestFileMask>(originBitBoard, notOtherPieceMask);
    const std::uint64_t southWestControlledSquares =
            fillRayWithMask<-9, kNotEastFileMask>(originBitBoard, notOtherPieceMask);
    const std::uint64_t northWestControlledSquares =
            fillRayWithMask<7, kNotEastFileMask>(originBitBoard, notOtherPieceMask);

    return (BitBoard)(northEastControlledSquares | southEastControlledSquares
                      | southWestControlledSquares | northWestControlledSquares);
}

BitBoard computeBishopXRaySquares(const BoardPosition origin, BitBoard anyPiece) {
    anyPiece &= ~origin;

    const std::uint64_t northEastOccupancy = (std::uint64_t)anyPiece & getFullRay(origin, 1, 1);
    const std::uint64_t southEastOccupancy = (std::uint64_t)anyPiece & getFullRay(origin, 1, -1);
    const std::uint64_t southWestOccupancy = (std::uint64_t)anyPiece & getFullRay(origin, -1, -1);
    const std::uint64_t northWestOccupancy = (std::uint64_t)anyPiece & getFullRay(origin, -1, 1);

    const int firstNEOccupied = std::countr_zero(northEastOccupancy);
    const int firstSEOccupied = kSquares - 1 - std::countl_zero(southEastOccupancy);
    const int firstSWOccupied = kSquares - 1 - std::countl_zero(southWestOccupancy);
    const int firstNWOccupied = std::countr_zero(northWestOccupancy);

    for (const auto& position :
         {firstNEOccupied, firstSEOccupied, firstSWOccupied, firstNWOccupied}) {
        if (position < 0 || position >= kSquares) {
            continue;
        }
        anyPiece &= ~(BoardPosition)position;
    }

    const std::uint64_t originBitBoard       = 1ULL << (int)origin;
    const std::uint64_t notBlockingPieceMask = ~((std::uint64_t)anyPiece);

    const std::uint64_t northEastXRay =
            fillRayWithMask<9, kNotWestFileMask>(originBitBoard, notBlockingPieceMask);
    const std::uint64_t southEastXRay =
            fillRayWithMask<-7, kNotWestFileMask>(originBitBoard, notBlockingPieceMask);
    const std::uint64_t southWestXRay =
            fillRayWithMask<-9, kNotEastFileMask>(originBitBoard, notBlockingPieceMask);
    const std::uint64_t northWestXRay =
            fillRayWithMask<7, kNotEastFileMask>(originBitBoard, notBlockingPieceMask);

    const std::uint64_t originBit = 1ULL << (int)origin;

    return (BitBoard)(northEastXRay | southEastXRay | southWestXRay | northWestXRay | originBit);
}

BitBoard computeRookControlledSquares(const BoardPosition origin, const BitBoard anyPiece) {
    const std::uint64_t originBitBoard    = 1ULL << (int)origin;
    const std::uint64_t notOtherPieceMask = ~((std::uint64_t)anyPiece);

    const std::uint64_t northControlledSquares =
            fillRayNoMask<8>(originBitBoard, notOtherPieceMask);
    const std::uint64_t eastControlledSquares =
            fillRayWithMask<1, kNotWestFileMask>(originBitBoard, notOtherPieceMask);
    const std::uint64_t southControlledSquares =
            fillRayNoMask<-8>(originBitBoard, notOtherPieceMask);
    const std::uint64_t westControlledSquares =
            fillRayWithMask<-1, kNotEastFileMask>(originBitBoard, notOtherPieceMask);

    return (BitBoard)(northControlledSquares | eastControlledSquares | southControlledSquares
                      | westControlledSquares);
}

BitBoard computeRookXRaySquares(const BoardPosition origin, BitBoard anyPiece) {
    anyPiece &= ~origin;

    const std::uint64_t northOccupancy = (std::uint64_t)anyPiece & getFullRay(origin, 0, 1);
    const std::uint64_t eastOccupancy  = (std::uint64_t)anyPiece & getFullRay(origin, 1, 0);
    const std::uint64_t southOccupancy = (std::uint64_t)anyPiece & getFullRay(origin, 0, -1);
    const std::uint64_t westOccupancy  = (std::uint64_t)anyPiece & getFullRay(origin, -1, 0);

    const int firstNorthOccupied = std::countr_zero(northOccupancy);
    const int firstEastOccupied  = std::countr_zero(eastOccupancy);
    const int firstSouthOccupied = kSquares - 1 - std::countl_zero(southOccupancy);
    const int firstWestOccupied  = kSquares - 1 - std::countl_zero(westOccupancy);

    for (const auto& position :
         {firstNorthOccupied, firstEastOccupied, firstSouthOccupied, firstWestOccupied}) {
        if (position < 0 || position >= kSquares) {
            continue;
        }
        anyPiece &= ~(BoardPosition)position;
    }

    const std::uint64_t originBitBoard       = 1ULL << (int)origin;
    const std::uint64_t notBlockingPieceMask = ~((std::uint64_t)anyPiece);

    const std::uint64_t northXRay = fillRayNoMask<8>(originBitBoard, notBlockingPieceMask);
    const std::uint64_t eastXRay =
            fillRayWithMask<1, kNotWestFileMask>(originBitBoard, notBlockingPieceMask);
    const std::uint64_t southXRay = fillRayNoMask<-8>(originBitBoard, notBlockingPieceMask);
    const std::uint64_t westXRay =
            fillRayWithMask<-1, kNotEastFileMask>(originBitBoard, notBlockingPieceMask);

    const std::uint64_t originBit = 1ULL << (int)origin;

    return (BitBoard)(northXRay | eastXRay | southXRay | westXRay | originBit);
}

constexpr BitBoard computeKingControlledSquares(const BoardPosition origin) {
    const auto [file, rank] = fileRankFromPosition(origin);

    BitBoard controlledSquares = BitBoard::Empty;
    for (int deltaFile = -1; deltaFile <= 1; ++deltaFile) {
        for (int deltaRank = -1; deltaRank <= 1; ++deltaRank) {
            const int newFile = file + deltaFile;
            const int newRank = rank + deltaRank;

            if (newFile < 0 || newFile >= kFiles || newRank < 0 || newRank >= kRanks) {
                continue;
            }
            const BoardPosition newPosition = positionFromFileRank(newFile, newRank);
            controlledSquares |= newPosition;
        }
    }
    controlledSquares &= ~origin;
    return controlledSquares;
}

constexpr std::array<BitBoard, kSquares> kKingControlledSquares = []() {
    std::array<BitBoard, kSquares> kingControlledSquares = {};
    for (int square = 0; square < kSquares; ++square) {
        kingControlledSquares[square] = computeKingControlledSquares((BoardPosition)square);
    }
    return kingControlledSquares;
}();

constexpr int kNumRookAttackEntries = 6 * 6 * (1 << 10) + 4 * 6 * (1 << 11) + 4 * (1 << 12);
constexpr int kNumBishopAttackEntries =
        4 * (1 << 9) + 12 * (1 << 7) + 20 * (1 << 5) + 4 * 6 * (1 << 5) + 4 * (1 << 6);

struct PackedRookAttacks {
    std::array<std::uint16_t, kNumRookAttackEntries> attacks;
    std::array<std::uint16_t*, kSquares> entries;
    std::array<std::uint64_t, kSquares> depositMasks;
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::array<std::uint64_t, kSquares> gRookLookupExtractMasks;  // NB: shared for x-rays and attacks

struct PackedBishopAttacks {
    std::array<std::uint16_t, kNumBishopAttackEntries> attacks;
    std::array<std::uint16_t*, kSquares> entries;
    std::array<std::uint64_t, kSquares> depositMasks;
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::array<std::uint64_t, kSquares> gBishopLookupExtractMasks;  // NB: shared for x-rays and attacks

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PackedRookAttacks gPackedRookAttacks;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PackedRookAttacks gPackedRookXRays;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PackedBishopAttacks gPackedBishopAttacks;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PackedBishopAttacks gPackedBishopXRays;

int calculatePackedRookAttacks() {
    int entriesCalculated = 0;

    for (int square = 0; square < kSquares; ++square) {
        const BoardPosition position = (BoardPosition)square;
        const auto [file, rank]      = fileRankFromPosition(position);

        const std::uint64_t north = getFullRay(position, 0, 1);
        const std::uint64_t east  = getFullRay(position, 1, 0);
        const std::uint64_t south = getFullRay(position, 0, -1);
        const std::uint64_t west  = getFullRay(position, -1, 0);

        const std::uint64_t squareBit = 1ULL << square;
        const std::uint64_t fullSight = (north | east | south | west) & ~squareBit;

        gPackedRookAttacks.depositMasks[square] = fullSight;

        std::uint64_t lookUpMask = fullSight;
        if (file > 0) {
            lookUpMask &= kNotWestFileMask;
        }
        if (file < 7) {
            lookUpMask &= kNotEastFileMask;
        }
        if (rank > 0) {
            lookUpMask &= kNotSouthRankMask;
        }
        if (rank < 7) {
            lookUpMask &= kNotNorthRankMask;
        }
        gRookLookupExtractMasks[square] = lookUpMask;

        MY_ASSERT(entriesCalculated < kNumRookAttackEntries);
        gPackedRookAttacks.entries[square] = &gPackedRookAttacks.attacks[entriesCalculated];

        const int numLookUpEntries = 1 << std::popcount(lookUpMask);
        for (int lookUpIndex = 0; lookUpIndex < numLookUpEntries; ++lookUpIndex) {
            const std::uint64_t lookUpBitBoard = pdep((std::uint64_t)lookUpIndex, lookUpMask);

            const BitBoard occupancy   = (BitBoard)lookUpBitBoard;
            const BitBoard rookAttacks = computeRookControlledSquares(position, occupancy);

            std::uint16_t packedRookAttacks =
                    (std::uint16_t)pext((std::uint64_t)rookAttacks, fullSight);

            MY_ASSERT(entriesCalculated < kNumRookAttackEntries);
            gPackedRookAttacks.attacks[entriesCalculated++] = packedRookAttacks;
        }
    }

    MY_ASSERT(entriesCalculated == kNumRookAttackEntries);

    return entriesCalculated;
}

int calculatePackedRookXRays() {
    int entriesCalculated = 0;

    for (int square = 0; square < kSquares; ++square) {
        const BoardPosition position = (BoardPosition)square;
        const auto [file, rank]      = fileRankFromPosition(position);

        const std::uint64_t north = getFullRay(position, 0, 1);
        const std::uint64_t east  = getFullRay(position, 1, 0);
        const std::uint64_t south = getFullRay(position, 0, -1);
        const std::uint64_t west  = getFullRay(position, -1, 0);

        const std::uint64_t squareBit = 1ULL << square;
        const std::uint64_t fullSight = (north | east | south | west);

        gPackedRookXRays.depositMasks[square] = fullSight;

        std::uint64_t lookUpMask = fullSight & ~squareBit;
        if (file > 0) {
            lookUpMask &= kNotWestFileMask;
        }
        if (file < 7) {
            lookUpMask &= kNotEastFileMask;
        }
        if (rank > 0) {
            lookUpMask &= kNotSouthRankMask;
        }
        if (rank < 7) {
            lookUpMask &= kNotNorthRankMask;
        }
        gRookLookupExtractMasks[square] = lookUpMask;

        MY_ASSERT(entriesCalculated < kNumRookAttackEntries);
        gPackedRookXRays.entries[square] = &gPackedRookXRays.attacks[entriesCalculated];

        const int numLookUpEntries = 1 << std::popcount(lookUpMask);
        for (int lookUpIndex = 0; lookUpIndex < numLookUpEntries; ++lookUpIndex) {
            const std::uint64_t lookUpBitBoard = pdep((std::uint64_t)lookUpIndex, lookUpMask);

            const BitBoard occupancy = (BitBoard)lookUpBitBoard;
            const BitBoard rookXRays = computeRookXRaySquares(position, occupancy);

            std::uint16_t packedRookXRays =
                    (std::uint16_t)pext((std::uint64_t)rookXRays, fullSight);

            MY_ASSERT(entriesCalculated < kNumRookAttackEntries);
            gPackedRookXRays.attacks[entriesCalculated++] = packedRookXRays;
        }
    }

    MY_ASSERT(entriesCalculated == kNumRookAttackEntries);

    return entriesCalculated;
}

int calculatePackedBishopAttacks() {
    int entriesCalculated = 0;

    for (int square = 0; square < kSquares; ++square) {
        const BoardPosition position = (BoardPosition)square;
        const auto [file, rank]      = fileRankFromPosition(position);

        const std::uint64_t northEast = getFullRay(position, 1, 1);
        const std::uint64_t southEast = getFullRay(position, 1, -1);
        const std::uint64_t southWest = getFullRay(position, -1, -1);
        const std::uint64_t northWest = getFullRay(position, -1, 1);

        const std::uint64_t squareBit = 1ULL << square;
        const std::uint64_t fullSight =
                (northEast | southEast | southWest | northWest) & ~squareBit;

        gPackedBishopAttacks.depositMasks[square] = fullSight;

        std::uint64_t lookUpMask = fullSight;
        if (file > 0) {
            lookUpMask &= kNotWestFileMask;
        }
        if (file < 7) {
            lookUpMask &= kNotEastFileMask;
        }
        if (rank > 0) {
            lookUpMask &= kNotSouthRankMask;
        }
        if (rank < 7) {
            lookUpMask &= kNotNorthRankMask;
        }
        gBishopLookupExtractMasks[square] = lookUpMask;

        MY_ASSERT(entriesCalculated < kNumBishopAttackEntries);
        gPackedBishopAttacks.entries[square] = &gPackedBishopAttacks.attacks[entriesCalculated];

        const int numLookUpEntries = 1 << std::popcount(lookUpMask);
        for (int lookUpIndex = 0; lookUpIndex < numLookUpEntries; ++lookUpIndex) {
            const std::uint64_t lookUpBitBoard = pdep((std::uint64_t)lookUpIndex, lookUpMask);

            const BitBoard occupancy     = (BitBoard)lookUpBitBoard;
            const BitBoard bishopAttacks = computeBishopControlledSquares(position, occupancy);

            std::uint16_t packedBishopAttacks =
                    (std::uint16_t)pext((std::uint64_t)bishopAttacks, fullSight);

            MY_ASSERT(entriesCalculated < kNumBishopAttackEntries);
            gPackedBishopAttacks.attacks[entriesCalculated++] = packedBishopAttacks;
        }
    }

    MY_ASSERT(entriesCalculated == kNumBishopAttackEntries);

    return entriesCalculated;
}

int calculatePackedBishopXRays() {
    int entriesCalculated = 0;

    for (int square = 0; square < kSquares; ++square) {
        const BoardPosition position = (BoardPosition)square;
        const auto [file, rank]      = fileRankFromPosition(position);

        const std::uint64_t northEast = getFullRay(position, 1, 1);
        const std::uint64_t southEast = getFullRay(position, 1, -1);
        const std::uint64_t southWest = getFullRay(position, -1, -1);
        const std::uint64_t northWest = getFullRay(position, -1, 1);

        const std::uint64_t squareBit = 1ULL << square;
        const std::uint64_t fullSight = (northEast | southEast | southWest | northWest);

        gPackedBishopXRays.depositMasks[square] = fullSight;

        std::uint64_t lookUpMask = fullSight & ~squareBit;
        if (file > 0) {
            lookUpMask &= kNotWestFileMask;
        }
        if (file < 7) {
            lookUpMask &= kNotEastFileMask;
        }
        if (rank > 0) {
            lookUpMask &= kNotSouthRankMask;
        }
        if (rank < 7) {
            lookUpMask &= kNotNorthRankMask;
        }
        gBishopLookupExtractMasks[square] = lookUpMask;

        MY_ASSERT(entriesCalculated < kNumBishopAttackEntries);
        gPackedBishopXRays.entries[square] = &gPackedBishopXRays.attacks[entriesCalculated];

        const int numLookUpEntries = 1 << std::popcount(lookUpMask);
        for (int lookUpIndex = 0; lookUpIndex < numLookUpEntries; ++lookUpIndex) {
            const std::uint64_t lookUpBitBoard = pdep((std::uint64_t)lookUpIndex, lookUpMask);

            const BitBoard occupancy   = (BitBoard)lookUpBitBoard;
            const BitBoard bishopXRays = computeBishopXRaySquares(position, occupancy);

            std::uint16_t packedBishopXRays =
                    (std::uint16_t)pext((std::uint64_t)bishopXRays, fullSight);

            MY_ASSERT(entriesCalculated < kNumBishopAttackEntries);
            gPackedBishopXRays.attacks[entriesCalculated++] = packedBishopXRays;
        }
    }

    MY_ASSERT(entriesCalculated == kNumBishopAttackEntries);

    return entriesCalculated;
}

const int dummyRookAttacks = calculatePackedRookAttacks();
const int dummyRookXRays   = calculatePackedRookXRays();

const int dummyBishopAttacks = calculatePackedBishopAttacks();
const int dummyBishopXRays   = calculatePackedBishopXRays();

template <typename PackedAttacksT>
FORCE_INLINE BitBoard getSliderAttack(
        const BoardPosition position,
        const BitBoard occupancy,
        const PackedAttacksT& packedAttacks,
        const std::array<std::uint64_t, kSquares>& lookupExtractMasks) {
    const std::uint64_t lookUpIndex =
            pext((std::uint64_t)occupancy, lookupExtractMasks[(int)position]);
    const std::uint16_t packedAttack = packedAttacks.entries[(int)position][lookUpIndex];
    return (BitBoard)pdep((std::uint64_t)packedAttack, packedAttacks.depositMasks[(int)position]);
}

}  // namespace

FORCE_INLINE BitBoard getPawnControlledSquares(const BoardPosition position, const Side side) {
    return getPawnControlledSquares(BitBoard::Empty | position, side);
}

FORCE_INLINE BitBoard getPawnControlledSquares(const BitBoard pawnBitBoard, const Side side) {
    auto leftForwardShift = [=](const BitBoard bitBoard) FORCE_INLINE {
        return side == Side::White ? (BitBoard)(((std::uint64_t)bitBoard & kNotWestFileMask) << 7)
                                   : (BitBoard)(((std::uint64_t)bitBoard & kNotWestFileMask) >> 9);
    };
    auto rightForwardShift = [=](const BitBoard bitBoard) FORCE_INLINE {
        return side == Side::White ? (BitBoard)(((std::uint64_t)bitBoard & kNotEastFileMask) << 9)
                                   : (BitBoard)(((std::uint64_t)bitBoard & kNotEastFileMask) >> 7);
    };

    const BitBoard leftCaptures  = leftForwardShift(pawnBitBoard);
    const BitBoard rightCaptures = rightForwardShift(pawnBitBoard);

    return leftCaptures | rightCaptures;
}

BitBoard getKingControlledSquares(const BoardPosition position) {
    return kKingControlledSquares[(int)position];
}

FORCE_INLINE BitBoard getRookAttack(const BoardPosition position, const BitBoard occupancy) {
    return getSliderAttack(position, occupancy, gPackedRookAttacks, gRookLookupExtractMasks);
}

FORCE_INLINE BitBoard getRookXRay(const BoardPosition position, const BitBoard occupancy) {
    return getSliderAttack(position, occupancy, gPackedRookXRays, gRookLookupExtractMasks);
}

FORCE_INLINE BitBoard getBishopAttack(const BoardPosition position, const BitBoard occupancy) {
    return getSliderAttack(position, occupancy, gPackedBishopAttacks, gBishopLookupExtractMasks);
}

FORCE_INLINE BitBoard getBishopXRay(const BoardPosition position, const BitBoard occupancy) {
    return getSliderAttack(position, occupancy, gPackedBishopXRays, gBishopLookupExtractMasks);
}

FORCE_INLINE std::uint64_t getFullRay(
        const BoardPosition position, const int fileIncrement, const int rankIncrement) {
    return kFullRays[fileIncrement + 1ULL][rankIncrement + 1ULL][(int)position];
}

FORCE_INLINE BitBoard getPieceControlledSquares(
        const Piece piece, const BoardPosition position, const BitBoard anyPiece) {
    switch (piece) {
        case Piece::Pawn:
            // Shouldn't call this function for pawns; call computePawnControlledSquares
            UNREACHABLE;
        case Piece::Knight:
            return kKnightControlledSquares[(int)position];
        case Piece::Bishop:
            return getBishopAttack(position, anyPiece);
        case Piece::Rook:
            return getRookAttack(position, anyPiece);
        case Piece::Queen: {
            const BitBoard bishopControlledSquares = getBishopAttack(position, anyPiece);
            const BitBoard rookControlledSquares   = getRookAttack(position, anyPiece);
            return bishopControlledSquares | rookControlledSquares;
        }
        case Piece::King:
            return getKingControlledSquares(position);
        default:
            UNREACHABLE;
    }
}

FORCE_INLINE BitBoard
getPieceXRays(const Piece piece, const BoardPosition position, const BitBoard anyPiece) {
    switch (piece) {
        case Piece::Pawn:
        case Piece::Knight:
            return BitBoard::Empty;

        case Piece::Bishop:
            return getBishopXRay(position, anyPiece);

        case Piece::Rook:
            return getRookXRay(position, anyPiece);

        case Piece::Queen: {
            const BitBoard bishopXRays = getBishopXRay(position, anyPiece);
            const BitBoard rookXRays   = getRookXRay(position, anyPiece);
            return bishopXRays | rookXRays;
        }

        case Piece::King:
            return BitBoard::Empty;

        default:
            UNREACHABLE;
    }
}
