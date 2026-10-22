import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "./AuthContext";
import { downloadReportRows, loadMyReportRows } from "./reportExport";

const value = (item) => String(item || "").trim();
const optionsFor = (rows, key) => [...new Set(rows.map((item) => value(item.data[key])).filter(Boolean))].sort((a, b) => a.localeCompare(b));
const asDate = (dateText) => {
  if (!dateText) return null;
  const parsed = new Date(dateText);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

export default function ReportManagementPage() {
  const navigate = useNavigate();
  const { appState, user, profile } = useAuth();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState({ status: "", club: "", position: "", scout: "", type: "", viewing: "", performance: "", potential: "", from: "", to: "" });

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    loadMyReportRows({ appState, user, profile })
      .then((rows) => { if (active) setReports(rows); })
      .catch((loadError) => { if (active) setError(loadError?.message || "Could not load your reports."); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [appState, user, profile]);

  const filteredReports = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    const from = asDate(filters.from);
    const to = asDate(filters.to);
    return reports.filter(({ data }) => {
      if (query && !Object.values(data).join(" ").toLocaleLowerCase().includes(query)) return false;
      if (filters.status && value(data.Status).toLocaleLowerCase() !== filters.status.toLocaleLowerCase()) return false;
      if (filters.club && data["Player's Club"] !== filters.club) return false;
      if (filters.position && data["Position Played"] !== filters.position) return false;
      if (filters.scout && data.Scout !== filters.scout) return false;
      if (filters.type && data["Report Type"] !== filters.type) return false;
      if (filters.viewing && data.Viewing !== filters.viewing) return false;
      if (filters.performance && data["Performance Grade"] !== filters.performance) return false;
      if (filters.potential && data["Potential Grade"] !== filters.potential) return false;
      const reportDate = asDate(data["Report Date"]);
      if (from && (!reportDate || reportDate < from)) return false;
      if (to && (!reportDate || reportDate > new Date(`${filters.to}T23:59:59`))) return false;
      return true;
    });
  }, [reports, search, filters]);

  const updateFilter = (key, nextValue) => setFilters((current) => ({ ...current, [key]: nextValue }));
  const clearFilters = () => { setSearch(""); setFilters({ status: "", club: "", position: "", scout: "", type: "", viewing: "", performance: "", potential: "", from: "", to: "" }); };
  const exportFiltered = (format) => {
    try {
      downloadReportRows(filteredReports, format);
      setNotice(`${filteredReports.length} filtered report${filteredReports.length === 1 ? "" : "s"} exported with all report fields.`);
      setError("");
    } catch (exportError) {
      setError(exportError?.message || "Could not export the filtered reports.");
      setNotice("");
    }
  };

  const statusOptions = optionsFor(reports, "Status");
  const clubOptions = optionsFor(reports, "Player's Club");
  const positionOptions = optionsFor(reports, "Position Played");
  const scoutOptions = optionsFor(reports, "Scout");
  const typeOptions = optionsFor(reports, "Report Type");
  const viewingOptions = optionsFor(reports, "Viewing");
  const performanceOptions = optionsFor(reports, "Performance Grade");
  const potentialOptions = optionsFor(reports, "Potential Grade");

  return (
    <main className="sr-page sr-report-management">
      <div className="sr-heading">
        <div>
          <button type="button" className="sr-outline sr-report-management-back" onClick={() => navigate("/scouting-reports")}>← Back to Reports &amp; Assignments</button>
          <p className="sr-eyebrow">REPORTS</p>
          <h1>Report Management</h1>
          <p>Review your reports, filter the ones you need, then export the matching reports to Excel or CSV.</p>
        </div>
      </div>

      {error && <div className="sr-report-management-message error" role="alert">{error}</div>}
      {notice && <div className="sr-report-management-message success" role="status">{notice}</div>}

      <section className="sr-report-management-panel" aria-label="Report filters">
        <div className="sr-report-management-toolbar">
          <label className="sr-report-management-search">
            <span>Search reports</span>
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Player, club, fixture, report text…" />
          </label>
          <div className="sr-report-management-export">
            <span>{loading ? "Loading reports…" : `${filteredReports.length} of ${reports.length} reports match`}</span>
            <button type="button" className="sr-cyan" disabled={loading || !filteredReports.length} onClick={() => exportFiltered("xlsx")}>Export filtered to Excel</button>
            <button type="button" className="sr-outline" disabled={loading || !filteredReports.length} onClick={() => exportFiltered("csv")}>Export filtered to CSV</button>
          </div>
        </div>

        <div className="sr-report-management-filters">
          <label><span>Status</span><select value={filters.status} onChange={(event) => updateFilter("status", event.target.value)}><option value="">All statuses</option>{statusOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label><span>Player’s club</span><select value={filters.club} onChange={(event) => updateFilter("club", event.target.value)}><option value="">All clubs</option>{clubOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label><span>Position</span><select value={filters.position} onChange={(event) => updateFilter("position", event.target.value)}><option value="">All positions</option>{positionOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label><span>Scout</span><select value={filters.scout} onChange={(event) => updateFilter("scout", event.target.value)}><option value="">All scouts</option>{scoutOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label><span>Report type</span><select value={filters.type} onChange={(event) => updateFilter("type", event.target.value)}><option value="">All report types</option>{typeOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label><span>Viewing</span><select value={filters.viewing} onChange={(event) => updateFilter("viewing", event.target.value)}><option value="">All viewing types</option>{viewingOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label><span>Performance grade</span><select value={filters.performance} onChange={(event) => updateFilter("performance", event.target.value)}><option value="">All grades</option>{performanceOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label><span>Potential grade</span><select value={filters.potential} onChange={(event) => updateFilter("potential", event.target.value)}><option value="">All grades</option>{potentialOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label><span>From date</span><input type="date" value={filters.from} onChange={(event) => updateFilter("from", event.target.value)} /></label>
          <label><span>To date</span><input type="date" value={filters.to} onChange={(event) => updateFilter("to", event.target.value)} /></label>
          <button type="button" className="sr-outline sr-report-management-clear" onClick={clearFilters}>Clear filters</button>
        </div>
      </section>

      <section className="sr-report-management-panel sr-report-management-list" aria-label="Your reports">
        {loading ? <p className="sr-report-management-empty">Loading your reports…</p> : filteredReports.length === 0 ? <p className="sr-report-management-empty">No reports match these filters.</p> : (
          <div className="sr-report-management-table-wrap">
            <table>
              <thead><tr><th>Player</th><th>Club</th><th>Position</th><th>Date</th><th>Fixture(s)</th><th>Scout</th><th>Type</th><th>Viewing</th><th>Grades</th><th>Status</th><th>Full report</th></tr></thead>
              <tbody>{filteredReports.map(({ id, data }) => (
                <ReportRow key={id} id={id} data={data} />
              ))}</tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}

function ReportRow({ id, data }) {
  const detailFields = ["Player", "Player's Club", "Position Played", "Report Date", "Fixture(s)", "Scout", "Report Type", "Viewing", "Footage", "Preferred Foot", "Performance Grade", "Potential Grade", "Reasons Why", "In Possession", "Out of Possession", "Physical", "On-Pitch Behaviour", "Strengths", "Weaknesses", "Conclusion", "Status"];
  return <>
    <tr>
      <td><strong>{data.Player || "Unnamed player"}</strong></td>
      <td>{data["Player's Club"] || "—"}</td>
      <td>{data["Position Played"] || "—"}</td>
      <td>{data["Report Date"] || "—"}</td>
      <td>{data["Fixture(s)"] || "—"}</td>
      <td>{data.Scout || "—"}</td>
      <td>{data["Report Type"] || "—"}</td>
      <td>{data.Viewing || "—"}</td>
      <td>{[data["Performance Grade"], data["Potential Grade"]].filter(Boolean).join(" / ") || "—"}</td>
      <td>{data.Status || "—"}</td>
      <td><details className="sr-report-management-details"><summary>View</summary><div className="sr-report-management-detail-grid">{detailFields.map((field) => <div key={`${id}-${field}`}><strong>{field}</strong><p>{data[field] || "—"}</p></div>)}</div></details></td>
    </tr>
  </>;
}
