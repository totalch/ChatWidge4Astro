/**
 * ingest.mjs — Vectorize docs into Cloudflare Vectorize (no external DB needed)
 * Run once locally: node scripts/ingest.mjs
 * Requires: npm install pdf-parse
 *
 * Text is stored inside Vectorize metadata — retrieved directly at query time.
 * No D1 / KV / Supabase required.
 */

import fs from 'fs'
import path from 'path'
import { createRequire } from 'module'

const require = createRequire(import.meta.url)
const pdfParse = require('pdf-parse')

// ─── Config ───────────────────────────────────────────────────
const CF_ACCOUNT_ID = process.env.CF_ACCOUNT_ID   // wrangler whoami
const CF_API_TOKEN  = process.env.CF_API_TOKEN    // Workers AI: Edit permission
const INDEX_NAME    = 'dogbro-docs'
const DOCS_DIR      = './docs'
const CHUNK_SIZE    = 600   // chars per chunk; 500–800 suits English prose
const BATCH_SIZE    = 50    // vectors per API call
// ──────────────────────────────────────────────────────────────

/** Split long text into overlapping chunks at sentence boundaries */
function chunkText(text, size = CHUNK_SIZE) {
  const clean = text
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/\f/g, '\n')   // PDF page breaks
    .trim()

  const chunks = []
  let current = ''
  const segments = clean.split(/(?<=[.!?\n])\s+/)

  for (const seg of segments) {
    if ((current + seg).length > size && current.length > 0) {
      chunks.push(current.trim())
      // one-sentence overlap for context continuity
      const overlap = current.split(/[.!?\n]/).slice(-1)[0] || ''
      current = overlap + ' ' + seg
    } else {
      current += (current ? ' ' : '') + seg
    }
  }
  if (current.trim()) chunks.push(current.trim())
  return chunks.filter(c => c.length > 30)
}

/** Embed using Cloudflare AI — bge-base-en-v1.5 (free, 768-dim) */
async function getEmbeddings(texts) {
  const res = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/ai/run/@cf/baai/bge-base-en-v1.5`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CF_API_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: texts }),
    }
  )
  if (!res.ok) throw new Error(`Embedding error: ${await res.text()}`)
  const data = await res.json()
  if (!data.result?.data) throw new Error('Unexpected embedding response')
  return data.result.data  // number[][]
}

/** Upsert vectors into Cloudflare Vectorize via REST API */
async function upsertVectors(vectors) {
  const ndjson = vectors.map(v => JSON.stringify(v)).join('\n')
  const res = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/vectorize/v2/indexes/${INDEX_NAME}/upsert`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CF_API_TOKEN}`,
        'Content-Type': 'application/x-ndjson',
      },
      body: ndjson,
    }
  )
  if (!res.ok) throw new Error(`Vectorize upsert error: ${await res.text()}`)
  return res.json()
}

/** Read a file and return plain text */
async function readFile(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  if (ext === '.txt' || ext === '.md') return fs.readFileSync(filePath, 'utf-8')
  if (ext === '.pdf') {
    const data = await pdfParse(fs.readFileSync(filePath))
    return data.text
  }
  console.warn(`  ⚠  Skipping unsupported format: ${filePath}`)
  return null
}

async function main() {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) {
    console.error('❌  Set environment variables first:')
    console.error('    export CF_ACCOUNT_ID=xxx CF_API_TOKEN=yyy')
    process.exit(1)
  }

  if (!fs.existsSync(DOCS_DIR)) {
    console.error(`❌  Docs directory not found: ${DOCS_DIR}`)
    process.exit(1)
  }

  const files = fs.readdirSync(DOCS_DIR)
    .filter(f => ['.txt', '.md', '.pdf'].includes(path.extname(f).toLowerCase()))

  if (!files.length) {
    console.error('❌  No txt / md / pdf files found in docs/')
    process.exit(1)
  }

  console.log(`📂  Found ${files.length} file(s). Starting ingestion...\n`)
  console.log(`ℹ   Upsert overwrites vectors with the same ID — safe to re-run.\n`)

  let totalChunks = 0

  for (const file of files) {
    console.log(`📄  Processing: ${file}`)
    const text = await readFile(path.join(DOCS_DIR, file))
    if (!text) continue

    const chunks = chunkText(text)
    console.log(`    Split into ${chunks.length} chunks`)

    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      const batch = chunks.slice(i, i + BATCH_SIZE)
      process.stdout.write(
        `    Embedding ${i + 1}–${Math.min(i + BATCH_SIZE, chunks.length)} / ${chunks.length}...`
      )

      const embeddings = await getEmbeddings(batch)

      const vectors = batch.map((chunk, j) => ({
        id: `${file.replace(/\W/g, '_')}_${i + j}`,
        values: embeddings[j],
        metadata: {
          text: chunk,          // ← stored here, returned at query time — no extra DB
          source: file,
          chunk_index: i + j,
        },
      }))

      await upsertVectors(vectors)
      console.log(' ✓')

      // avoid hitting rate limits
      if (i + BATCH_SIZE < chunks.length) await new Promise(r => setTimeout(r, 400))
    }

    totalChunks += chunks.length
    console.log(`    ✅  Done\n`)
  }

  console.log(`🎉  Ingestion complete — ${totalChunks} vectors stored in "${INDEX_NAME}"`)
  console.log(`    Text lives in Vectorize metadata. No external DB needed.`)
}

main().catch(err => { console.error('❌ ', err.message); process.exit(1) })
