import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import fetch from "node-fetch";
import { parse } from "csv-parse/sync";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CSV_PATH = path.join(__dirname, "player-photos.csv");
const OUTPUT_DIR = path.join(__dirname, "../public/player-photos");

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

const csv = fs.readFileSync(CSV_PATH, "utf8");
const records = parse(csv, { columns: true, skip_empty_lines: true });

console.log(`Saving images to: ${OUTPUT_DIR}`);
console.log(`Total players: ${records.length}`);

let count = 0;

for (const player of records) {
  const url = player["Photo URL"];
  const name = (player["Player Name"] || "unknown")
    .replace(/[^a-z0-9]/gi, "_")
    .toLowerCase();

  if (!url) continue;

  const filePath = path.join(OUTPUT_DIR, `${name}.png`);

  // ⛔ skip already downloaded
  if (fs.existsSync(filePath)) continue;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Fetch failed");

    const buffer = Buffer.from(await res.arrayBuffer());
    fs.writeFileSync(filePath, buffer);

    count++;
    if (count % 100 === 0) {
      console.log(`Downloaded ${count} images...`);
    }

    // 🧘 tiny delay to avoid freezing / rate limit
    await new Promise(r => setTimeout(r, 25));

  } catch {
    console.log(`✖ Failed: ${name}`);
  }
}

console.log("Done.");
