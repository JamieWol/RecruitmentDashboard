import { parseExcelSheetRows, parseWordHtml } from "./reportImport";

describe("old report import parsing", () => {
  test("maps spreadsheet columns and normalises UK match dates", () => {
    const [record] = parseExcelSheetRows([
      ["Player Name", "Player Club", "Position Played", "Opposition", "Match Date", "Preferred Foot", "Performance Grade", "Strengths", "Weaknesses", "Conclusion"],
      ["Alex Example", "Example FC", "CM", "Example United", "06/10/2026", "Right", "4", "Keeps the ball well", "Needs to scan earlier", "A composed midfielder."],
    ]);

    expect(record).toMatchObject({
      player: "Alex Example",
      club: "Example FC",
      position: "CM",
      game: "Example United",
      date: "2026-10-06",
      report: {
        type: "Long Report",
        playedPosition: "CM",
        foot: "Right",
        performance: "4",
        strengths: "Keeps the ball well",
        weaknesses: "Needs to scan earlier",
        conclusion: "A composed midfielder.",
      },
    });
  });

  test("splits Word content by player headings and keeps section text", () => {
    const records = parseWordHtml([
      "<p><strong>A. Example</strong></p><p>Example FC</p>",
      "<p><strong>Strengths</strong></p><p>Presses well</p>",
      "<p><strong>Weaknesses</strong></p><p>Needs to be stronger</p>",
      "<p><strong>B. Example</strong></p><p>Other FC</p>",
      "<p><strong>Conclusion</strong></p><p>Good technical profile</p>",
    ].join(""));

    expect(records).toHaveLength(2);
    expect(records[0]).toMatchObject({ player: "A. Example", club: "Example FC", report: { strengths: "Presses well", weaknesses: "Needs to be stronger" } });
    expect(records[1]).toMatchObject({ player: "B. Example", club: "Other FC", report: { conclusion: "Good technical profile" } });
  });

  test("recognises the existing player, club, fixture, date and notes layout", () => {
    const [record] = parseWordHtml([
      "<p><strong>D. KOWNACKI</strong></p><p>Werder Bremen</p>",
      "<p><strong>FORTUNA DÜSSELDORF v HERTHA BSC</strong></p><p>22/03/2026</p>",
      "<p>Works hard out of possession</p><p>Needs to be stronger in duels</p>",
    ].join(""));

    expect(record).toMatchObject({
      player: "D. KOWNACKI",
      club: "Werder Bremen",
      game: "FORTUNA DÜSSELDORF v HERTHA BSC",
      date: "2026-03-22",
      report: { conclusion: "Works hard out of possession\nNeeds to be stronger in duels" },
    });
  });
});
