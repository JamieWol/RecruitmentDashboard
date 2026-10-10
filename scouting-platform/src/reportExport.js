import * as XLSX from "xlsx";
import { supabase } from "./supabaseClient";

const EXPORT_COLUMNS = [
  "Player", "Player's Club", "Position Played", "Report Date", "Fixture(s)", "Scout",
  "Report Type", "Viewing", "Footage", "Preferred Foot", "Performance Grade",
  "Potential Grade", "Reasons Why", "In Possession", "Out of Possession", "Physical",
  "On-Pitch Behaviour", "Strengths", "Weaknesses", "Conclusion", "Status",
];

const textValue = (value) => {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.map(textValue).filter(Boolean).join("\n");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value).trim();
};

const contentFields = ["playedPosition", "performance", "potential", "conclusion", "reasons", "inPossession", "outPossession", "physical", "behaviour", "strengths", "weaknesses"];
const hasReportContent = (record) => contentFields.some((key) => textValue(record?.report?.[key]));

const belongsToUser = (record, user, profile) => {
  const ids = [user?.id, profile?.id].filter(Boolean).map(String);
  const names = [profile?.full_name, user?.email].filter(Boolean).map((name) => String(name).trim().toLowerCase());
  const assignedId = record?.scoutId ?? record?.scout_id;
  if (assignedId && ids.includes(String(assignedId))) return true;
  const scout = String(record?.scout || "").trim().toLowerCase();
  if (scout && names.includes(scout)) return true;
  // Legacy rows have no assignment owner metadata; appState itself is account scoped.
  return !assignedId && !scout;
};

const fixtureText = (record, report) => {
  const fixtures = [record?.games, report?.__fixtures].find((list) => Array.isArray(list) && list.length) || [];
  if (fixtures.length) {
    return fixtures.map((fixture, index) => {
      const name = typeof fixture === "string" ? fixture : fixture?.name || fixture?.fixture || "";
      const date = (typeof fixture === "object" ? fixture?.date : "") || report?.__fixtureDates?.[index] || record?.fixtureDates?.[index] || "";
      return [date, name].filter(Boolean).join(" · ");
    }).filter(Boolean).join(" | ");
  }
  return textValue(record?.fixture_summary || record?.game || "");
};

const rowFor = (record, profile) => {
  const report = record?.report || {};
  return {
    "Player": textValue(record?.player || record?.Name || record?.name),
    "Player's Club": textValue(record?.player_club || record?.playerClub || record?.club || record?.Club),
    "Position Played": textValue(report.playedPosition || record?.position),
    "Report Date": textValue(record?.completed_at || record?.completedAt || record?.date || report.date),
    "Fixture(s)": fixtureText(record, report),
    "Scout": textValue(record?.scout || profile?.full_name),
    "Report Type": textValue(report.type),
    "Viewing": textValue(report.viewing || record?.viewing),
    "Footage": textValue(report.footage || record?.footage),
    "Preferred Foot": textValue(report.foot),
    "Performance Grade": textValue(report.performance || record?.performance),
    "Potential Grade": textValue(report.potential || record?.potential),
    "Reasons Why": textValue(report.reasons),
    "In Possession": textValue(report.inPossession),
    "Out of Possession": textValue(report.outPossession),
    "Physical": textValue(report.physical),
    "On-Pitch Behaviour": textValue(report.behaviour),
    "Strengths": textValue(report.strengths),
    "Weaknesses": textValue(report.weaknesses),
    "Conclusion": textValue(report.conclusion),
    "Status": textValue(record?.status || "Draft"),
  };
};

const reportKey = (record) => String(record?.assignment_id ?? record?.id ?? [record?.player, record?.completed_at || record?.date, record?.fixture_summary].join("|"));

export async function exportMyReports({ appState, user, profile, format }) {
  if (!user?.id) throw new Error("Sign in to export your reports.");

  const ownAssignments = (appState?.assignments || []).filter((record) => hasReportContent(record) && belongsToUser(record, user, profile));
  const { data, error } = await supabase.from("club_reports").select("*").eq("author_id", user.id).eq("status", "Published");
  if (error) throw error;

  const merged = new Map();
  ownAssignments.forEach((record) => merged.set(reportKey(record), record));
  (data || []).filter(hasReportContent).forEach((record) => {
    const key = reportKey(record);
    const local = merged.get(key);
    merged.set(key, local ? { ...local, ...record, report: { ...(local.report || {}), ...(record.report || {}) } } : record);
  });

  const rows = [...merged.values()].map((record) => rowFor(record, profile));
  if (!rows.length) throw new Error("No saved reports were found for your account.");

  const dateStamp = new Date().toISOString().slice(0, 10);
  const worksheet = XLSX.utils.json_to_sheet(rows, { header: EXPORT_COLUMNS });
  worksheet["!cols"] = EXPORT_COLUMNS.map((column) => ({ wch: Math.min(48, Math.max(16, column.length + 2)) }));
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "My Reports");
  if (format === "csv") {
    const csv = XLSX.utils.sheet_to_csv(worksheet);
    const blob = new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `scoutpro-my-reports-${dateStamp}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } else {
    XLSX.writeFile(workbook, `scoutpro-my-reports-${dateStamp}.xlsx`);
  }
  return rows.length;
}
