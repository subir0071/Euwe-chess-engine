#pragma once

#include "BoardPosition.h"
#include "Piece.h"

#include <format>

#include <cstdint>

enum class MoveFlags : std::uint8_t {
    None = 0,
    // Lowest 3 bits: Piece if promoting
    IsCapture   = 1 << 3,
    IsEnPassant = 1 << 4,
    IsCastle    = 1 << 5,
};

[[nodiscard]] constexpr MoveFlags operator|(const MoveFlags lhs, const MoveFlags rhs) {
    return static_cast<MoveFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

[[nodiscard]] constexpr MoveFlags operator|(const MoveFlags flags, const Piece promotionPiece) {
    return static_cast<MoveFlags>(static_cast<int>(flags) | static_cast<int>(promotionPiece));
}

constexpr MoveFlags& operator|=(MoveFlags& lhs, const MoveFlags rhs) {
    lhs = lhs | rhs;
    return lhs;
}

constexpr MoveFlags& operator|=(MoveFlags& flags, const Piece promotionPiece) {
    flags = flags | promotionPiece;
    return flags;
}

[[nodiscard]] constexpr bool operator&(const MoveFlags lhs, const MoveFlags rhs) {
    return (static_cast<int>(lhs) & static_cast<int>(rhs)) != 0;
}

[[nodiscard]] constexpr Piece getPromotionPiece(const MoveFlags flags) {
    return static_cast<Piece>((int)flags & 7);
}

[[nodiscard]] constexpr bool isPromotion(const MoveFlags flags) {
    return getPromotionPiece(flags) != Piece::Pawn;
}

[[nodiscard]] constexpr bool isQueenPromotion(const MoveFlags flags) {
    return getPromotionPiece(flags) == Piece::Queen;
}

[[nodiscard]] constexpr bool isCapture(const MoveFlags flags) {
    return flags & MoveFlags::IsCapture;
}

[[nodiscard]] constexpr bool isCaptureOrQueenPromo(const MoveFlags flags) {
    return isCapture(flags) || isQueenPromotion(flags);
}

[[nodiscard]] constexpr bool isEnPassant(const MoveFlags flags) {
    return flags & MoveFlags::IsEnPassant;
}

[[nodiscard]] constexpr bool isCastle(const MoveFlags flags) {
    return flags & MoveFlags::IsCastle;
}

class GameState;

struct Move {
    Piece pieceToMove  = Piece::Invalid;
    BoardPosition from = BoardPosition::Invalid;
    BoardPosition to   = BoardPosition::Invalid;
    MoveFlags flags    = MoveFlags::None;

    bool operator==(const Move& other) const = default;

    [[nodiscard]] std::string toAlgebraic(const GameState& gameState) const;
    [[nodiscard]] std::string toUci() const;

    // Long algebraic notation for moves, except no indicators for check or checkmate
    // {piece}{from}-{to}
    // If capture: use 'x' instead of '-'
    // For promotions: suffix '={promotion piece}'
    // For en passant: suffix ' e.p.'
    // For castling: normal algebraic notation
    // Examples: Pe2-e4, Rd3xd7, Pe5xd6 e.p.
    [[nodiscard]] std::string toExtendedString() const;

    [[nodiscard]] static Move fromAlgebraic(std::string_view algebraic, const GameState& gameState);

    [[nodiscard]] static Move fromUci(std::string_view uci, const GameState& gameState);
};

void doBasicSanityChecks(const Move& move, const GameState& gameState);

[[nodiscard]] constexpr Piece getPromotionPiece(const Move& move) {
    return getPromotionPiece(move.flags);
}

[[nodiscard]] constexpr bool isPromotion(const Move& move) {
    return isPromotion(move.flags);
}

[[nodiscard]] constexpr bool isQueenPromotion(const Move& move) {
    return isQueenPromotion(move.flags);
}

[[nodiscard]] constexpr bool isCapture(const Move& move) {
    return isCapture(move.flags);
}

[[nodiscard]] constexpr bool isCaptureOrQueenPromo(const Move& move) {
    return isCaptureOrQueenPromo(move.flags);
}

[[nodiscard]] constexpr bool isEnPassant(const Move& move) {
    return isEnPassant(move.flags);
}

[[nodiscard]] constexpr bool isCastle(const Move& move) {
    return isCastle(move.flags);
}
