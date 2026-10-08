import { findImportedPlayerMatch, parseExcelSheetRows, parseWordHtml, splitImportedFixtures } from "./reportImport";

describe("old report import parsing", () => {
  test("splits a multi-game imported report into separately dated fixtures", () => {
    expect(splitImportedFixtures(
      "FORTUNA DÜSSELDORF v HERTHA BSC 22/03/2026 | HERTHA BSC v BOCHUM 14/03/2026 | HERTHA BSC v FREIBURG 10/02/2026",
      "2026-03-22",
    )).toEqual([
      { name: "FORTUNA DÜSSELDORF v HERTHA BSC", date: "2026-03-22" },
      { name: "HERTHA BSC v BOCHUM", date: "2026-03-14" },
      { name: "HERTHA BSC v FREIBURG", date: "2026-02-10" },
    ]);
  });

  test("matches a unique player despite an abbreviated first name and avoids ambiguous matches", () => {
    expect(findImportedPlayerMatch("Y. Ohashi", [
      { id: 7, Name: "Yuki Ohashi" },
      { id: 8, Name: "Other Player" },
    ])).toMatchObject({ id: 7, Name: "Yuki Ohashi" });
    expect(findImportedPlayerMatch("Yuki Ohashi", [
      { id: 7, Name: "Yuki Ohashi" },
      { id: 9, Name: "Yuki Ohashi" },
    ])).toBeNull();
  });

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

  test("reads Game(s) and separates adjacent table sections into their correct fields", () => {
    const [record] = parseWordHtml(`<table>
      <tr><td><p>Player</p></td><td><p>Y. Ohashi</p></td><td><p>Club</p></td><td><p>Lommel</p></td></tr>
      <tr><td><p>Game(s)</p></td><td><p>BLACKBURN ROVERS v COVENTRY CITY 17/04/2026</p><p>BLACKBURN ROVERS v MILLWALL 10/04/2026</p></td></tr>
      <tr><td><p>Reason why</p></td><td><p>Technically good striker.</p></td></tr>
      <tr><td><p>Strengths</p><p>Good first touch.</p><p>Runs in behind well.</p><p>Weaknesses</p><p>Needs to be more clinical.</p></td></tr>
      <tr><td><p>Physical</p><p>Competes well in duels.</p><p>On Pitch Behaviour</p><p>Works hard for the team.</p></td></tr>
    </table>`);

    expect(record).toMatchObject({
      player: "Y. Ohashi",
      club: "Lommel",
      game: "BLACKBURN ROVERS v COVENTRY CITY 17/04/2026\nBLACKBURN ROVERS v MILLWALL 10/04/2026",
      report: {
        reasons: "Technically good striker.",
        strengths: "Good first touch.\nRuns in behind well.",
        weaknesses: "Needs to be more clinical.",
        physical: "Competes well in duels.",
        behaviour: "Works hard for the team.",
      },
    });
  });

  test("recognises decorated section titles and keeps each section in its own field", () => {
    const [record] = parseWordHtml([
      "<p><strong>Yuki Ohashi</strong></p><p>Lommel</p>",
      "<p><strong>Weaknesses</strong></p><p>Needs to improve his finishing.</p>",
      "<p><strong>Strengths points</strong></p><p>Good first touch and intelligent movement.</p>",
      "<p><strong>Physical points</strong></p><p>Competes well in physical duels.</p>",
      "<p><strong>On-pitch behaviour&#x20;</strong></p><p>Works hard for the team.</p>",
      "<p><strong>Conclusion (75 words)</strong></p><p>A hard-working striker with good movement.</p>",
    ].join(""));

    expect(record.report).toMatchObject({
      weaknesses: "Needs to improve his finishing.",
      strengths: "Good first touch and intelligent movement.",
      physical: "Competes well in physical duels.",
      behaviour: "Works hard for the team.",
      conclusion: "A hard-working striker with good movement.",
    });
  });

  test("keeps side-by-side Word table sections in their own columns", () => {
    const [record] = parseWordHtml(`<table>
      <tr><td><p>Player</p></td><td><p>Yuki Ohashi</p></td><td><p>Club</p></td><td><p>Lommel</p></td></tr>
      <tr><td><p>Strengths</p></td><td><p>Weaknesses</p></td></tr>
      <tr><td><p>Finishes well.</p><p>Good attacking movement.</p></td><td><p>Needs to be more clinical.</p><p>Not dominant aerially.</p></td></tr>
      <tr><td><p>Physical</p></td><td><p>On-pitch behaviour&#x20;</p></td></tr>
      <tr><td><p>Competes well in duels.</p></td><td><p>Works hard for the team.</p></td></tr>
      <tr><td><p>Conclusion (75 words)</p></td><td><p>A hard-working striker.</p></td></tr>
    </table>`);

    expect(record).toMatchObject({
      player: "Yuki Ohashi",
      club: "Lommel",
      report: {
        strengths: "Finishes well.\nGood attacking movement.",
        weaknesses: "Needs to be more clinical.\nNot dominant aerially.",
        physical: "Competes well in duels.",
        behaviour: "Works hard for the team.",
        conclusion: "A hard-working striker.",
      },
    });
  });

  test("keeps each game date with its fixture when dates are listed separately", () => {
    const [record] = parseWordHtml(`<table>
      <tr><td><p>Player</p></td><td><p>Yuki Ohashi</p></td></tr>
      <tr><td><p>Game(s)</p></td><td><p>BLACKBURN ROVERS v COVENTRY CITY 22/04/2026</p><p>SHEFFIELD UNITED v BLACKBURN ROVERS 15/04/2026</p></td></tr>
    </table>`);

    expect(record).toMatchObject({
      game: "BLACKBURN ROVERS v COVENTRY CITY 22/04/2026\nSHEFFIELD UNITED v BLACKBURN ROVERS 15/04/2026",
    });
    expect(splitImportedFixtures(record.game, record.date)).toEqual([
      { name: "BLACKBURN ROVERS v COVENTRY CITY", date: "2026-04-22" },
      { name: "SHEFFIELD UNITED v BLACKBURN ROVERS", date: "2026-04-15" },
    ]);
  });

  test("splits adjacent fixtures when Word export collapses their line breaks", () => {
    expect(splitImportedFixtures(
      "BLACKBURN ROVERS v COVENTRY CITY BLACKBURN ROVERS v LEICESTER CITY SHEFFIELD UNITED v BLACKBURN ROVERS 22/04/2026",
      "2026-05-02",
    )).toEqual([
      { name: "BLACKBURN ROVERS v COVENTRY CITY", date: "2026-05-02" },
      { name: "BLACKBURN ROVERS v LEICESTER CITY", date: "" },
      { name: "SHEFFIELD UNITED v BLACKBURN ROVERS", date: "2026-04-22" },
    ]);
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
