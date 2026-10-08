import { clubBadgePathFor } from "./clubBadgePaths";

describe("Italian club badge aliases", () => {
  test.each(["Inter", "Inter Milan", "Inter Milano", "Internazionale", "Internazionale Milano", "FC Internazionale Milano"])("maps %s to Inter", (name) => {
    expect(clubBadgePathFor(name)).toBe("logos/italy/inter.png");
  });

  test.each(["Milan", "AC Milan", "AC Milano"])("maps %s to AC Milan", (name) => {
    expect(clubBadgePathFor(name)).toBe("logos/italy/milan.png");
  });
});

describe("Bayern Munich badge names", () => {
  test.each(["Bayern Munich", "FC Bayern Munich", "Bayern München", "FC Bayern München", "Bayern Munchen"])("maps %s to the Bayern badge", (name) => {
    expect(clubBadgePathFor(name)).toBe("logos/germany/bayern-munchen.png");
  });
});
