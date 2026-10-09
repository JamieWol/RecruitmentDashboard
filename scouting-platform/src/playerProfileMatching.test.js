import { choosePlayerRecord, normalizePlayerName, playerNameMatchScore } from "./playerProfileMatching";

test("matches full player names against abbreviated database names", () => {
  expect(playerNameMatchScore("Tobias Bech Kristensen", "T. Bech Kristensen")).toBeGreaterThan(0);
  expect(normalizePlayerName("Tobias Bech Kristensen")).toBe("tobias bech kristensen");
});

test("uses the player's report club to select among duplicate database rows", () => {
  const rows = [
    { id: 1, Name: "Tobias Bech Kristensen", Team: "Former Club", Age: 24 },
    { id: 2, Name: "Tobias Bech Kristensen", Team: "AGF", DOB: "2002-01-01", Nationality: "Danish" },
  ];
  expect(choosePlayerRecord(rows, "Tobias Bech Kristensen", "AGF")).toMatchObject({
    id: 2,
    Team: "AGF",
    DOB: "2002-01-01",
  });
});

test("merges missing bio fields from duplicate rows for the same club", () => {
  const rows = [
    { id: 1, Name: "Tobias Bech Kristensen", Team: "AGF", Age: 24 },
    { id: 2, Name: "Tobias Bech Kristensen", Team: "AGF", DOB: "2002-01-01", Nationality: "Danish" },
  ];
  expect(choosePlayerRecord(rows, "Tobias Bech Kristensen", "AGF")).toMatchObject({
    Team: "AGF",
    DOB: "2002-01-01",
    Age: 24,
    Nationality: "Danish",
  });
});

test("leaves identical names at different clubs unmatched when no club context exists", () => {
  const rows = [
    { id: 1, Name: "Tobias Bech Kristensen", Team: "AGF" },
    { id: 2, Name: "Tobias Bech Kristensen", Team: "Former Club" },
  ];
  expect(choosePlayerRecord(rows, "Tobias Bech Kristensen")).toBeNull();
});
