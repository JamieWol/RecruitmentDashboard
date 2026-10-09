jest.mock("./supabaseClient", () => ({ supabase: { from: jest.fn() } }));

import { supabase } from "./supabaseClient";
import { migrateAndLoadState } from "./cloudState";

describe("account scoped squad plans", () => {
  beforeEach(() => {
    localStorage.clear();
    supabase.from.mockReset();
  });

  it("keeps unmatched browser-wide plans recoverable without exposing them to another account", async () => {
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
    expect(localStorage.getItem("squadPlans")).not.toBeNull();
    expect(localStorage.getItem("squadPlanClub")).toBe("Bradford City");
  });

  it("migrates a legacy plan when it matches the account's existing squad draft", async () => {
    const accountState = {
      user_id: "account-a",
      assignments: [],
      shortlists: [],
      tags: [],
      squadPlan: {
        club: "Bradford City",
        formation: "4-2-3-1",
        players: [{ id: "player-a" }],
        savedPlans: [],
      },
    };
    const maybeSingle = jest.fn().mockResolvedValue({ data: accountState, error: null });
    const eq = jest.fn(() => ({ maybeSingle }));
    const select = jest.fn(() => ({ eq }));
    const upsert = jest.fn().mockResolvedValue({ error: null });
    supabase.from.mockReturnValue({ select, upsert });
    localStorage.setItem("squadPlans", JSON.stringify([{ id: "plan-a", club: "Bradford City", formation: "4-2-3-1", players: [{ id: "player-a" }] }]));

    const state = await migrateAndLoadState({ id: "account-a" });

    expect(state.squadPlan.savedPlans).toHaveLength(1);
    expect(state.squadPlan.savedPlans[0].id).toBe("plan-a");
    expect(upsert).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem("squadPlans")).toBeNull();
  });
});
