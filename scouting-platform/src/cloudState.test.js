jest.mock("./supabaseClient", () => ({ supabase: { from: jest.fn() } }));

import { supabase } from "./supabaseClient";
import { migrateAndLoadState } from "./cloudState";

describe("account scoped squad plans", () => {
  beforeEach(() => {
    localStorage.clear();
    supabase.from.mockReset();
  });

  it("does not load another account's old browser saved plans", async () => {
    const accountState = {
      user_id: "account-b",
      assignments: [{ id: 2, player: "Account B player" }],
      shortlists: [],
      tags: [],
      squadPlan: { club: "Leeds United", savedPlans: [] },
    };
    const maybeSingle = jest.fn().mockResolvedValue({ data: accountState, error: null });
    const eq = jest.fn(() => ({ maybeSingle }));
    const select = jest.fn(() => ({ eq }));
    const upsert = jest.fn().mockResolvedValue({ error: null });
    supabase.from.mockReturnValue({ select, upsert });
    localStorage.setItem("squadPlans", JSON.stringify([{ id: "old", club: "Bradford City", players: [{ id: "player-a" }] }]));
    localStorage.setItem("squadPlanClub", "Bradford City");

    const state = await migrateAndLoadState({ id: "account-b" });

    expect(state.squadPlan).toEqual(accountState.squadPlan);
    expect(upsert).not.toHaveBeenCalled();
    expect(localStorage.getItem("squadPlans")).toBeNull();
    expect(localStorage.getItem("squadPlanClub")).toBeNull();
  });
});
