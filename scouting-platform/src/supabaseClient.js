const url =
  process.env.REACT_APP_SUPABASE_URL ||
  process.env.SUPABASE_URL ||
  "https://syjsmvvsvvprxibqoizw.supabase.co";
const key =
  process.env.REACT_APP_SUPABASE_PUBLISHABLE_KEY ||
  process.env.REACT_APP_SUPABASE_ANON_KEY ||
  process.env.SUPABASE_PUBLISHABLE_KEY ||
  process.env.SUPABASE_ANON_KEY ||
  "sb_publishable_iEKXNM7vpUypQ_nryAYhHA_7H577-I2";

export const supabase = {
  from: (table) => ({
    select: (columns) => ({
      ilike: (column, value) => ({
        limit: async (count) => {
          const params = new URLSearchParams({
            select: columns,
            [column]: `ilike.*${String(value).trim()}*`,
            limit: String(count),
          });
          const response = await fetch(
            `${url}/rest/v1/${table}?${params.toString()}`,
            { headers: { apikey: key, Authorization: `Bearer ${key}` } },
          );
          return {
            data: response.ok ? await response.json() : null,
            error: response.ok ? null : await response.text(),
          };
        },
      }),
      limit: async (count) => {
        const response = await fetch(
          `${url}/rest/v1/${table}?select=${encodeURIComponent(columns)}&limit=${count}`,
          { headers: { apikey: key, Authorization: `Bearer ${key}` } },
        );
        return {
          data: response.ok ? await response.json() : null,
          error: response.ok ? null : await response.text(),
        };
      },
    }),
  }),
};
