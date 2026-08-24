// Scan the built site for source syntax that failed to parse.
//
// Run automatically after `pnpm build` via the postbuild hook, and in CI.
// Exits non-zero on any finding.
//
// Why this exists: markup Docusaurus does not understand is not an error, it
// is *text*. `:::caution Title` (the Docusaurus v2 form, dropped in v3 in
// favour of `:::caution[Title]`) parsed as a plain paragraph on 13 pages for
// months, and nothing caught it -- not the build, not `onBrokenLinks: 'throw'`,
// not a link check. The whole class is invisible unless you look at the output.
//
// So: strip the places source syntax is legitimate (code blocks, inline code,
// scripts), then look for it in what is left. Anything found is markup the
// reader is seeing raw.

import {readFileSync, readdirSync, existsSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const buildDir = resolve(here, '..', process.argv[2] ?? 'build');

if (!existsSync(buildDir)) {
  console.error(`no build at ${buildDir} -- run \`pnpm build\` first`);
  process.exit(2);
}

// Where source syntax is legitimate, and so must not be scanned.
const STRIP = [
  /<pre\b[\s\S]*?<\/pre>/gi,
  /<code\b[\s\S]*?<\/code>/gi,
  /<script\b[\s\S]*?<\/script>/gi,
  /<style\b[\s\S]*?<\/style>/gi,
  /<noscript\b[\s\S]*?<\/noscript>/gi,
];

// Each check names markup that should never survive into visible text.
const CHECKS = [
  ['unparsed-admonition', /:::[a-z]*/g,
   'admonition directive rendered as text -- v3 needs :::type[Title], not :::type Title'],
  ['unparsed-bold', /\*\*[^*\n]{1,60}\*\*/g, 'bold markers rendered as text'],
  ['unparsed-link', /\[[^\]\n]{1,60}\]\([^)\n]{1,80}\)/g, 'link syntax rendered as text'],
  ['unparsed-heading', /(?:^|\n)\s*#{1,6}\s+\S/g, 'heading marker rendered as text'],
  ['unparsed-table-row', /(?:^|\n)\s*\|[^|\n]+\|/g, 'table row rendered as text'],
  ['doubled-list-marker', /(?:^|\n)\s*-\s+-\s+\S/g,
   'nested one-item list -- valid CommonMark, but indents the entry one level too deep'],
  ['visible-html-comment', /<!--/g, 'HTML comment visible to the reader'],
  ['jsx-brace-leak', /\{\s*[a-zA-Z_$][\w.]*\s*\}/g, 'JSX expression rendered as text'],
];

const htmlFiles = [];
(function walk(dir) {
  for (const e of readdirSync(dir, {withFileTypes: true})) {
    const p = join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.name.endsWith('.html')) htmlFiles.push(p);
  }
})(buildDir);

const unescape = (s) => s
  .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"')
  .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(+d))
  .replace(/&amp;/g, '&');

const findings = new Map();
for (const file of htmlFiles) {
  let text = readFileSync(file, 'utf-8');
  for (const re of STRIP) text = text.replace(re, ' ');
  // Keep tag boundaries as line breaks so the ^-anchored checks still work.
  text = unescape(text.replace(/<[^>]+>/g, '\n'));
  for (const [name, re, why] of CHECKS) {
    for (const m of text.matchAll(re)) {
      if (!findings.has(name)) findings.set(name, {why, hits: new Set()});
      findings.get(name).hits.add(`${relative(buildDir, file)}  ${m[0].replace(/\s+/g, ' ').trim().slice(0, 70)}`);
    }
  }
}

console.log(`checked ${htmlFiles.length} built pages`);
if (findings.size === 0) {
  console.log('no unparsed markup found');
  process.exit(0);
}

for (const [name, {why, hits}] of findings) {
  console.log(`\n${name}: ${why}`);
  for (const h of [...hits].sort()) console.log(`  ${h}`);
}
const total = [...findings.values()].reduce((n, f) => n + f.hits.size, 0);
console.log(`\n${total} finding(s). These are rendering as raw text on the site.`);
process.exit(1);
