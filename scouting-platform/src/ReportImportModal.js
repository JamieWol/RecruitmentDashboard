import React, { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { supabase } from "./supabaseClient";
import { findImportedPlayerMatch, normalizeImportedPlayerName, parseOldReportFiles } from "./reportImport";

const editableReportFields = [
  ["Conclusion", "conclusion"], ["Strengths", "strengths"], ["Weaknesses", "weaknesses"],
  ["Reasons why", "reasons"], ["In possession", "inPossession"],
  ["Out of possession", "outPossession"], ["Physical", "physical"], ["On-pitch behaviour", "behaviour"],
];
const gradeOptions = (values) => [<option key="" value="">Not included</option>, ...values.map((value) => <option key={value} value={value}>{value}</option>)];

const readImportDraft = (storageKey) => {
  try { return JSON.parse(sessionStorage.getItem(`${storageKey}:draft`) || "null") || {}; }
  catch { return {}; }
};

export default function ReportImportModal({ onClose, onImport, storageKey = "scoutingReportImport:guest" }) {
  const [records, setRecords] = useState(() => readImportDraft(storageKey).records || []);
  const [errors, setErrors] = useState(() => readImportDraft(storageKey).errors || []);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(() => readImportDraft(storageKey).message || "");
  const [expanded, setExpanded] = useState(() => readImportDraft(storageKey).expanded || {});
  const duplicateKeys = useMemo(() => {
    const counts = new Map();
    records.forEach((record) => {
      const key = [normalizeImportedPlayerName(record.player), record.date, normalizeImportedPlayerName(record.game)].join("|");
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    return counts;
  }, [records]);
  React.useEffect(() => {
    try { sessionStorage.setItem(`${storageKey}:draft`, JSON.stringify({ records, errors, message, expanded })); }
    catch { /* Keep the live import usable if browser session storage is full. */ }
  }, [records, errors, message, expanded, storageKey]);

  const updateRecord = (index, updates) => setRecords((current) => current.map((record, row) => row === index ? { ...record, ...updates } : record));
  const updateReport = (index, field, value) => setRecords((current) => current.map((record, row) => row === index ? { ...record, report: { ...record.report, [field]: value } } : record));

  const parseFiles = async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    setLoading(true);
    setMessage("");
    setErrors([]);
    try {
      const parsed = await parseOldReportFiles(files);
      const nextRecords = parsed.records;
      const matchOne = async (record) => {
        if (!record.player) return { matches: [] };
        try {
          let { data, error } = await supabase.from("players").select("*").ilike("Name", `%${record.player.trim()}%`).limit(20);
          if (error) return { matches: [], matchError: error.message };
          if (!data?.length) {
            const surname = record.player.trim().split(/\s+/).at(-1);
            if (surname && surname.length > 1) {
              const fallback = await supabase.from("players").select("*").ilike("Name", `%${surname}%`).limit(20);
              if (!fallback.error) data = fallback.data || [];
            }
          }
          const rows = data || [];
          const matches = rows.map((player) => ({ id: player.id, player_id: player.player_id, Name: player.Name, name: player.name, club: player.club, Club: player.Club, team: player.team, Team: player.Team }));
          const player = findImportedPlayerMatch(record.player, rows);
          if (player) {
            return { matches, playerId: player.id || player.player_id || "", player: player.Name || player.name || record.player, club: record.club || "", position: record.position || "" };
          }
          return { matches };
        } catch { return { matches: [] }; }
      };
      const matchResults = [];
      for (let offset = 0; offset < nextRecords.length; offset += 8) {
        matchResults.push(...await Promise.all(nextRecords.slice(offset, offset + 8).map(matchOne)));
      }
      setRecords(nextRecords.map((record, index) => ({ ...record, ...matchResults[index] })));
      setErrors(parsed.errors);
      if (!nextRecords.length && !parsed.errors.length) setMessage("No reports were found in those files.");
      else if (nextRecords.length) setMessage(`${nextRecords.length} report${nextRecords.length === 1 ? "" : "s"} ready to review.`);
    } catch (error) {
      setErrors([error?.message || "Could not read those files."]);
    } finally {
      setLoading(false);
      event.target.value = "";
    }
  };

  const choosePlayer = (index, value) => {
    if (value === "") {
      updateRecord(index, { playerId: "", playerMatch: null });
      return;
    }
    const record = records[index];
    const player = record.matches?.find((item) => String(item.id || item.player_id || item.Name || item.name) === value);
    if (!player) return;
    updateRecord(index, {
      playerId: player.id || player.player_id || "",
      playerMatch: player,
      player: player.Name || player.name || record.player,
      club: record.club || "",
      position: record.position || "",
    });
  };

  const submit = async () => {
    const valid = records.filter((record) => record.player.trim());
    if (!valid.length) { setMessage("Add at least one player name before importing."); return; }
    setSaving(true);
    try {
      await onImport(valid);
      sessionStorage.removeItem(`${storageKey}:draft`);
    } catch (error) {
      setMessage(error?.message || "The reports could not be saved. Please try again.");
      setSaving(false);
    }
  };

  return createPortal((
    <div className="sr-modal sr-import-modal" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="sr-form sr-import-form" role="dialog" aria-modal="true" aria-labelledby="sr-import-title">
        <div className="sr-form-head">
          <div><div className="sr-kicker">REPORT ARCHIVE</div><h2 id="sr-import-title">Import old reports</h2><p>Upload Word .docx or Excel files. Check the extracted details before saving them as drafts.</p></div>
          <button type="button" className="sr-close" onClick={onClose} aria-label="Close">×</button>
        </div>
        <label className="sr-import-drop">
          <input type="file" multiple accept=".docx,.xlsx,.xls,.csv" onChange={parseFiles} disabled={loading || saving} />
          <strong>{loading ? "Reading reports and checking players…" : "Choose Word or Excel files"}</strong>
          <span>Excel: one report per row, with a Player or Name column. Word: one report per file or reports with Player headings.</span>
          <small>Supported: .docx, .xlsx, .xls and .csv</small>
        </label>
        {message && <p className="sr-import-message" role="status">{message}</p>}
        {!!errors.length && <div className="sr-import-errors" role="alert">{errors.map((error) => <p key={error}>{error}</p>)}</div>}
        {!!records.length && <div className="sr-import-list">
          <div className="sr-import-list-head"><strong>{records.length} reports to review</strong><span>Nothing is saved until you import these drafts.</span></div>
          {records.map((record, index) => {
            const duplicateKey = [normalizeImportedPlayerName(record.player), record.date, normalizeImportedPlayerName(record.game)].join("|");
            const duplicate = Boolean(record.player && duplicateKeys.get(duplicateKey) > 1);
            return <article className="sr-import-card" key={`${record.sourceFile}-${record.sourceSheet || ""}-${index}`}>
              <div className="sr-import-card-top"><strong>{record.sourceFile}{record.sourceSheet ? ` · ${record.sourceSheet}` : ""}</strong><button type="button" className="sr-import-remove" onClick={() => setRecords((current) => current.filter((_, row) => row !== index))}>Remove</button></div>
              {duplicate && <p className="sr-import-warning">Possible duplicate player, fixture and date. Check before importing.</p>}
              <div className="sr-import-grid">
                <label className="sr-field"><span>Player</span><input value={record.player} onChange={(event) => updateRecord(index, { player: event.target.value, playerId: "" })} /></label>
                <label className="sr-field"><span>Match to player database</span><select value={record.playerId || ""} onChange={(event) => choosePlayer(index, event.target.value)}><option value="">Keep this name / choose a match</option>{(record.matches || []).map((player, matchIndex) => { const key = player.id || player.player_id || player.Name || player.name; return <option key={`${key}-${matchIndex}`} value={String(key)}>{player.Name || player.name}{player.club || player.Club || player.team ? ` · ${player.club || player.Club || player.team}` : ""}</option>; })}</select></label>
                {!!record.player && !record.playerId && <small className="sr-import-unmatched">No exact player selected. This report will stay attached to the name you entered.</small>}
                <label className="sr-field"><span>Club at the time</span><input value={record.club} onChange={(event) => updateRecord(index, { club: event.target.value })} /></label>
                <label className="sr-field"><span>Position</span><input value={record.position} onChange={(event) => updateRecord(index, { position: event.target.value, report: { ...record.report, playedPosition: event.target.value } })} /></label>
                <label className="sr-field"><span>Fixture / opponent</span><input value={record.game} onChange={(event) => updateRecord(index, { game: event.target.value })} /></label>
                <label className="sr-field"><span>Match date</span><input type="date" value={record.date || ""} onChange={(event) => updateRecord(index, { date: event.target.value })} /></label>
                <label className="sr-field"><span>Viewing</span><select value={record.viewing || "Video"} onChange={(event) => updateRecord(index, { viewing: event.target.value })}><option>Video</option><option>Live</option></select></label>
                <label className="sr-field"><span>Preferred foot</span><select value={record.report.foot || ""} onChange={(event) => updateReport(index, "foot", event.target.value)}>{gradeOptions(["Right", "Left", "Both"])}</select></label>
                <label className="sr-field"><span>Performance grade</span><select value={record.report.performance || ""} onChange={(event) => updateReport(index, "performance", event.target.value)}>{gradeOptions([5, 4, 3, 2, 1])}</select></label>
                <label className="sr-field"><span>Potential grade</span><select value={record.report.potential || ""} onChange={(event) => updateReport(index, "potential", event.target.value)}>{gradeOptions(["A", "B", "C", "D", "E", "F"])}</select></label>
              </div>
              <button type="button" className="sr-import-details-toggle" onClick={() => setExpanded((current) => ({ ...current, [index]: !current[index] }))}>{expanded[index] ? "Hide report content" : "Review and edit report content"}</button>
              {expanded[index] && <div className="sr-import-report-content">{editableReportFields.map(([label, field]) => <label className="sr-field" key={field}><span>{label}</span><textarea rows={field === "conclusion" ? 3 : 2} value={record.report[field] || ""} onChange={(event) => updateReport(index, field, event.target.value)} /></label>)}</div>}
            </article>;
          })}
        </div>}
        <div className="sr-actions sr-import-actions"><button type="button" className="sr-outline" onClick={onClose} disabled={saving}>Cancel</button><button type="button" className="sr-cyan" onClick={submit} disabled={!records.length || loading || saving}>{saving ? "Saving drafts…" : `Import ${records.length || ""} draft reports`}</button></div>
      </section>
    </div>
  ), document.body);
}
