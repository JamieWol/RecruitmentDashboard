import * as XLSX from "xlsx";
import mammoth from "mammoth";

export const importedReportFields = {
  player: ["player", "player name", "name", "scouted player", "player full name", "player being scouted", "scouted player name"],
  club: ["club", "current club", "team", "player club", "club name", "club at time", "team at time", "club team"],
  position: ["position", "playing position", "primary position", "position played", "player position", "role"],
  game: ["fixture", "fixtures", "fixture(s)", "game", "games", "game(s)", "match", "matches", "match(es)", "opposition", "opponent", "fixture game", "match fixture", "games watched", "matches watched", "opposition played", "teams played against"],
  date: ["date", "dates", "match date", "match date(s)", "game date", "game date(s)", "fixture date", "report date", "viewing date", "date watched"],
  viewing: ["viewing", "viewing type", "viewing method", "context", "live or video", "live video", "scouting context"],
  foot: ["foot", "preferred foot", "dominant foot", "strong foot"],
  performance: ["performance", "performance grade", "performance rating", "grade performance", "performance score", "performance grade 1 5"],
  potential: ["potential", "potential grade", "potential rating", "grade potential", "potential score", "potential grade a f"],
  reasons: ["reasons", "reason", "reasons why", "reason why", "grade reasons", "reasoning", "reason for grade", "reason for rating"],
  inPossession: ["in possession", "in possession notes", "in possession analysis", "on the ball", "with the ball", "attacking play", "technical analysis"],
  outPossession: ["out of possession", "out possession", "out of possession notes", "without the ball", "off the ball", "defensive notes", "defensive analysis"],
  physical: ["physical", "physical notes", "physical attributes", "athleticism"],
  behaviour: ["on pitch behaviour", "on pitch behavior", "on pitch attitude", "on pitch character", "behaviour", "behavior", "mentality"],
  strengths: ["strengths", "key strengths", "positive points", "positives", "strength", "what went well"],
  weaknesses: ["weaknesses", "areas to improve", "areas for improvement", "development areas", "development needs", "negatives", "weakness", "areas to develop"],
  conclusion: ["conclusion", "summary", "scout summary", "overall assessment", "overall report", "overall summary", "player summary", "evaluation"],
};

const clean = (value) => String(value ?? "").replace(/\u00a0/g, " ").trim();
const fixtureDateToISO = (value) => {
  const text = clean(value);
  const iso = text.match(/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$/);
  if (iso) return `${iso[1]}-${iso[2].padStart(2, "0")}-${iso[3].padStart(2, "0")}`;
  const uk = text.match(/^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$/);
  if (uk) return `${uk[3].length === 2 ? `20${uk[3]}` : uk[3]}-${uk[2].padStart(2, "0")}-${uk[1].padStart(2, "0")}`;
  return text;
};
export const splitImportedFixtures = (value, fallbackDate = "") => {
  const entries = clean(value).split(/\s*(?:\||\r?\n|;)+\s*/).map(clean).filter(Boolean);
  const fallbackDates = clean(fallbackDate).split(/\s*(?:\||\r?\n|;)+\s*/).map(fixtureDateToISO);
  return entries.map((entry, index) => {
    const match = entry.match(/\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[/. -]\d{1,2}[/. -]\d{2,4})\b/);
    const date = match ? fixtureDateToISO(match[1]) : fallbackDates[index] || fallbackDates[0] || "";
    const name = clean(match ? entry.replace(match[0], " ").replace(/[|·,–—-]+$/g, "") : entry).replace(/^[|·,–—-]+|[|·,–—-]+$/g, "");
    return name ? { name, date } : null;
  }).filter(Boolean);
};
const toISODate = (value) => {
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.toISOString().slice(0, 10);
  const text = clean(value);
  if (!text) return "";
  const iso = text.match(/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$/);
  if (iso) return `${iso[1]}-${iso[2].padStart(2, "0")}-${iso[3].padStart(2, "0")}`;
  const uk = text.match(/^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$/);
  if (uk) {
    const year = uk[3].length === 2 ? `20${uk[3]}` : uk[3];
    return `${year}-${uk[2].padStart(2, "0")}-${uk[1].padStart(2, "0")}`;
  }
  if (/^\d+(?:\.\d+)?$/.test(text)) {
    const parsed = XLSX.SSF.parse_date_code(Number(text));
    if (parsed?.y && parsed?.m && parsed?.d) return `${parsed.y}-${String(parsed.m).padStart(2, "0")}-${String(parsed.d).padStart(2, "0")}`;
  }
  const parsed = new Date(text);
  return Number.isNaN(parsed.getTime()) ? text : parsed.toISOString().slice(0, 10);
};
const normalizeLabel = (value) => clean(value).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
const aliasToField = new Map(Object.entries(importedReportFields).flatMap(([field, aliases]) => aliases.map((alias) => [normalizeLabel(alias), field])));
const getField = (label) => {
  const normalized = normalizeLabel(label);
  const exact = aliasToField.get(normalized);
  if (exact) return exact;
  const withoutParenthetical = normalizeLabel(String(label || "").replace(/\([^)]*\)/g, " "));
  const parentheticalMatch = aliasToField.get(withoutParenthetical);
  if (parentheticalMatch) return parentheticalMatch;
  const base = normalized
    .replace(/\s+\d+\s+(?:words?|bullets?|points?|items?|lines?).*$/i, "")
    .replace(/\s+(?:points?|bullet points?|comments?|observations?|evaluation|section)$/i, "")
    .trim();
  return aliasToField.get(base);
};
const reportFields = ["foot", "playedPosition", "performance", "potential", "conclusion", "reasons", "inPossession", "outPossession", "physical", "behaviour", "strengths", "weaknesses"];

const emptyRecord = (sourceFile) => ({
  player: "", club: "", position: "", game: "", date: "", viewing: "Video", sourceFile,
  report: Object.fromEntries(reportFields.map((field) => [field, ""])) ,
});

const normalizeRecord = (record, sourceFile) => {
  const report = { ...emptyRecord(sourceFile).report, ...(record.report || {}) };
  const position = record.position || report.playedPosition || "";
  if (!report.playedPosition && position) report.playedPosition = position;
  const hasLongSections = Boolean(report.inPossession || report.outPossession || report.physical || report.behaviour || report.strengths || report.weaknesses);
  report.type = hasLongSections ? "Long Report" : "Short Report";
  return { ...emptyRecord(sourceFile), ...record, position, date: toISODate(record.date), report };
};

const rowToRecord = (row, sourceFile) => {
  const fields = {};
  Object.entries(row).forEach(([key, value]) => {
    const field = getField(key);
    if (field) fields[field] = field === "date" ? toISODate(value) : clean(value);
  });
  const report = {};
  Object.entries(fields).forEach(([field, value]) => {
    if (reportFields.includes(field)) report[field] = value;
  });
  const record = normalizeRecord({
    player: fields.player || "", club: fields.club || "", position: fields.position || "",
    game: fields.game || "", date: fields.date || "", viewing: fields.viewing || "Video", report,
  }, sourceFile);
  if (!record.player) return null;
  return record;
};

const rowsFromSheet = (rows, sourceFile, sheetName) => {
  const headerIndex = rows.findIndex((row) => row.some((cell) => getField(cell) === "player"));
  if (headerIndex < 0) return [];
  const headers = rows[headerIndex].map((cell, index) => clean(cell) || `Column ${index + 1}`);
  return rows.slice(headerIndex + 1).map((row) => {
    if (!row.some((cell) => clean(cell))) return null;
    return rowToRecord(Object.fromEntries(headers.map((header, index) => [header, row[index]])), sourceFile);
  }).filter(Boolean).map((record) => ({ ...record, sourceSheet: sheetName }));
};
export const parseExcelSheetRows = (rows, sourceFile = "Workbook.xlsx", sheetName = "Sheet1") => rowsFromSheet(rows, sourceFile, sheetName);

const collectWordBlocks = (html) => {
  const documentNode = new DOMParser().parseFromString(html, "text/html");
  const blocks = [];
  const blockText = (node) => {
    const parts = [];
    const walk = (current) => {
      if (current.nodeType === 3) { parts.push(current.nodeValue || ""); return; }
      if (current.nodeType !== 1) return;
      if (current.tagName === "BR") { parts.push("\n"); return; }
      const isBlock = /^(P|DIV|LI|H[1-6])$/.test(current.tagName);
      if (isBlock && parts.length && !parts.at(-1).endsWith("\n")) parts.push("\n");
      Array.from(current.childNodes).forEach(walk);
      if (isBlock && !parts.at(-1)?.endsWith("\n")) parts.push("\n");
    };
    walk(node);
    return parts.join("").split(/\r?\n/).map(clean).filter(Boolean);
  };
  const addTextBlock = (node, text, tableCell = null) => {
    if (!text) return;
    const heading = /^H[1-6]$/.test(node.tagName);
    const strong = heading || Boolean(node.querySelector?.("strong,b")) || (node.tagName === "STRONG" || node.tagName === "B");
    blocks.push({ text, cells: null, strong, heading, tableCell });
  };
  const walk = (node) => {
    if (node.tagName === "TABLE") {
      const tableId = blocks.filter((block) => block.tableCell).at(-1)?.tableCell.tableId + 1 || 0;
      Array.from(node.querySelectorAll("tr")).forEach((row, rowIndex) => {
        Array.from(row.querySelectorAll("th,td")).forEach((cell, columnIndex) => {
          const tableCell = { tableId, rowIndex, columnIndex };
          const paragraphs = Array.from(cell.querySelectorAll("p,h1,h2,h3,h4,h5,h6,li"));
          if (paragraphs.length) {
            paragraphs.forEach((paragraph) => blockText(paragraph).forEach((text) => addTextBlock(paragraph, text, tableCell)));
          } else {
            blockText(cell).forEach((text) => blocks.push({ text, cells: null, strong: cell.children.length === 0 && Boolean(cell.querySelector("strong,b")), heading: false, tableCell }));
          }
        });
      });
      return;
    }
    if (/^(P|H[1-6]|LI)$/.test(node.tagName)) {
      blockText(node).forEach((text) => addTextBlock(node, text));
      return;
    }
    const children = Array.from(node.children || []);
    if (children.length) children.forEach(walk);
    else blockText(node).forEach((text) => addTextBlock(node, text));
  };
  Array.from(documentNode.body.children).forEach(walk);
  return blocks;
};

const parseWordBlock = (block) => {
  if (block.cells?.length > 1) {
    const direct = getField(block.cells[0]);
    if (direct) return { field: direct, value: block.cells.slice(1).filter(Boolean).join(" | ") };
  }
  const text = clean(block.text);
  const divider = text.match(/^([^:|–—-]{2,45})\s*(?::|\||–|—|\s-\s)\s*(.*)$/);
  if (divider) {
    const field = getField(divider[1]);
    if (field) return { field, value: clean(divider[2]) };
  }
  const field = getField(text.replace(/[:：]$/, ""));
  if (field) return { field, value: "" };
  return null;
};

const likelyPlayerHeading = (block) => {
  if (!block.strong && !block.heading) return false;
  const text = clean(block.text);
  if (!text || text.length > 64 || getField(text) || /\b(?:v|vs|versus)\b/i.test(text) || /^\d{1,2}[/. -]\d{1,2}[/. -]\d{2,4}$/.test(text)) return false;
  const words = text.split(/\s+/);
  return words.length >= 2 && words.length <= 5 && /[a-z]/i.test(text) && /^[\p{L}\p{M}.'’\-\s]+$/u.test(text);
};

const wordBlocksToRecords = (blocks, sourceFile) => {
  const groups = [];
  let current = [];
  blocks.forEach((block) => {
    const parsed = parseWordBlock(block);
    const beginsNamedRecord = parsed?.field === "player" && parsed.value && current.some((item) => parseWordBlock(item)?.field === "player");
    const beginsHeadingRecord = likelyPlayerHeading(block) && current.some((item) => parseWordBlock(item)?.field === "player" || likelyPlayerHeading(item));
    if (current.length && (beginsNamedRecord || beginsHeadingRecord)) {
      groups.push(current);
      current = [];
    }
    current.push(block);
  });
  if (current.length) groups.push(current);

  return groups.map((group) => {
    const record = emptyRecord(sourceFile);
    const looseNotes = [];
    let activeField = "";
    let sawPlayer = false;
    const tableActiveFields = new Map();
    const tableRowFields = new Map();
    const appendFieldValue = (field, value) => {
      const text = clean(value);
      if (!text) return;
      if (field === "player") {
        if (!record.player) { record.player = text; sawPlayer = true; }
      } else if (["club", "position", "game", "date", "viewing"].includes(field)) {
        record[field] = [record[field], field === "date" ? toISODate(text) : text].filter(Boolean).join("\n");
      } else {
        const reportKey = field === "position" ? "playedPosition" : field;
        record.report[reportKey] = [record.report[reportKey], text].filter(Boolean).join("\n");
      }
    };
    const tableFieldFor = (tableCell) => {
      if (!tableCell) return "";
      const { tableId, rowIndex, columnIndex } = tableCell;
      const rowFields = tableRowFields.get(`${tableId}:${rowIndex}`);
      const sameCellField = rowFields?.get(columnIndex);
      if (sameCellField) return sameCellField;
      const earlier = rowFields
        ? Array.from(rowFields.entries()).filter(([column]) => column < columnIndex).sort((a, b) => b[0] - a[0])
        : [];
      return earlier[0]?.[1] || tableActiveFields.get(`${tableId}:${columnIndex}`) || "";
    };
    group.forEach((block, index) => {
      const parsed = parseWordBlock(block);
      if (parsed) {
        if (block.tableCell) {
          const { tableId, rowIndex, columnIndex } = block.tableCell;
          tableActiveFields.set(`${tableId}:${columnIndex}`, parsed.field);
          const rowKey = `${tableId}:${rowIndex}`;
          if (!tableRowFields.has(rowKey)) tableRowFields.set(rowKey, new Map());
          tableRowFields.get(rowKey).set(columnIndex, parsed.field);
        } else activeField = parsed.field;
        if (parsed.field === "player") {
          if (parsed.value) appendFieldValue(parsed.field, parsed.value);
          return;
        }
        if (parsed.value) appendFieldValue(parsed.field, parsed.value);
        return;
      }
      const text = clean(block.text);
      if (!text) return;
      if (likelyPlayerHeading(block) && !record.player) {
        record.player = text;
        sawPlayer = true;
        activeField = "";
        return;
      }
      const field = block.tableCell ? tableFieldFor(block.tableCell) : activeField;
      if (field) {
        appendFieldValue(field, text);
        if (!block.tableCell && field === "player") activeField = "";
        return;
      }
      if (sawPlayer && !record.club && index <= 3) record.club = text;
      else if (sawPlayer) {
        const dateMatch = text.match(/\b\d{1,2}[/. -]\d{1,2}[/. -]\d{2,4}\b/);
        if (dateMatch) {
          record.date = toISODate(dateMatch[0]);
          const fixtureText = clean(text.replace(dateMatch[0], "").replace(/[|·,–—-]+$/g, ""));
          if (/\b(?:v|vs|versus)\b/i.test(fixtureText)) record.game = fixtureText;
          return;
        }
        if (/\b(?:v|vs|versus)\b/i.test(text) && !record.game) record.game = text;
        else looseNotes.push(text);
      }
    });
    if (!reportFields.some((field) => record.report[field]) && looseNotes.length) record.report.conclusion = looseNotes.join("\n");
    const result = normalizeRecord(record, sourceFile);
    return result.player ? result : null;
  }).filter(Boolean);
};
export const parseWordHtml = (html, sourceFile = "Report.docx") => wordBlocksToRecords(collectWordBlocks(html), sourceFile);

const parseWordFile = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  const { value: html } = await mammoth.convertToHtml({ arrayBuffer });
  const records = parseWordHtml(html, file.name);
  const filenamePlayer = clean(file.name.replace(/\.docx$/i, "")).replace(/[_]+/g, " ").trim();
  if (records.length === 1 && !records[0].player && /^[\p{L}\p{M}.'’\-\s]+$/u.test(filenamePlayer)) records[0].player = filenamePlayer;
  return records;
};

export async function parseOldReportFiles(files) {
  const records = [];
  const errors = [];
  for (const file of Array.from(files || [])) {
    try {
      const extension = file.name.split(".").pop().toLowerCase();
      if (extension === "docx") {
        const parsed = await parseWordFile(file);
        if (!parsed.length) errors.push(`${file.name}: no player report was recognised. Add a Player/Name heading or upload one report per file.`);
        records.push(...parsed);
      } else if (["xlsx", "xls", "csv"].includes(extension)) {
        const workbook = XLSX.read(await file.arrayBuffer(), { type: "array", cellDates: true });
        const parsed = workbook.SheetNames.flatMap((sheetName) => {
          const rows = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName], { header: 1, defval: "", raw: false });
          return rowsFromSheet(rows, file.name, sheetName);
        });
        if (!parsed.length) errors.push(`${file.name}: no rows with a Player/Name column were recognised.`);
        records.push(...parsed);
      } else {
        errors.push(`${file.name}: use Word .docx, Excel .xlsx/.xls, or CSV.`);
      }
    } catch (error) {
      errors.push(`${file.name}: ${error?.message || "could not be read"}`);
    }
  }
  return { records, errors };
}

export const normalizeImportedPlayerName = (value) => clean(value).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
export const findImportedPlayerMatch = (name, candidates = []) => {
  const query = normalizeImportedPlayerName(name);
  if (!query) return null;
  const queryTokens = query.split(" ");
  const ranked = candidates.map((player) => {
    const candidateName = normalizeImportedPlayerName(player.Name || player.name || player.full_name || player.fullName);
    const candidateTokens = candidateName.split(" ");
    let score = 0;
    if (candidateName === query) score = 100;
    else if (queryTokens.length >= 2 && queryTokens.every((token) => candidateTokens.includes(token))) score = 95;
    else if (queryTokens.length >= 2 && candidateTokens.length >= 2) {
      const sameOrder = queryTokens[0] === candidateTokens[0] && queryTokens.at(-1) === candidateTokens.at(-1);
      const reversedOrder = queryTokens[0] === candidateTokens.at(-1) && queryTokens.at(-1) === candidateTokens[0];
      const sameSurname = queryTokens.at(-1) === candidateTokens.at(-1) || reversedOrder;
      const sameFirst = sameOrder || reversedOrder;
      const initialMatch = queryTokens[0].length === 1 && candidateTokens[0].startsWith(queryTokens[0]);
      if (sameSurname && (sameFirst || initialMatch)) score = 90;
    }
    return { player, score };
  }).filter((entry) => entry.score >= 90);
  if (!ranked.length) return null;
  const bestScore = Math.max(...ranked.map((entry) => entry.score));
  const best = ranked.filter((entry) => entry.score === bestScore);
  return best.length === 1 ? best[0].player : null;
};
