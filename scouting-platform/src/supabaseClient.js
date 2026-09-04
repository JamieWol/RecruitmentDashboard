const url =
  process.env.REACT_APP_SUPABASE_URL ||
  "https://syjsmvvsvvprxibqoizw.supabase.co";
const key =
  process.env.REACT_APP_SUPABASE_ANON_KEY ||
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzIiwicmVmIjoic3lqc21tdnZzdnZwcnhpYnFvaXp3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODMzMzIxMDAsImV4cCI6MjA5ODkwODEwMH0.V2pn7hcNTa8hvWpm2FH8pG2hrymVs8W8TojdCzGaeGE";

export const supabase = {
  from: (table) => ({
    select: (columns) => ({
      limit: async (count) => {
        const response = await fetch(
          `${url}/rest/v1/${table}?select=${encodeURIComponent(columns)}&limit=${count}`,
          {
            headers: { apikey: key, Authorization: `Bearer ${key}` },
          },
        );
        return {
          data: response.ok ? await response.json() : null,
          error: response.ok ? null : await response.text(),
        };
      },
    }),
  }),
};
