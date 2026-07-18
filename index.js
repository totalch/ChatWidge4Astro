/**
 * worker/index.js — Cloudflare Worker RAG chatbot
 * Flow: user message → Workers AI embed → Vectorize query → Gemini stream
 *
 * Bindings in wrangler.toml:
 *   [ai]         binding = "AI"
 *   [[vectorize]] binding = "VECTORIZE"  index_name = "dogbro-docs"
 *
 * Secrets (wrangler secret put):
 *   GEMINI_API_KEY   — aistudio.google.com/apikey  (free)
 *   ALLOWED_ORIGIN   — https://dogbro.com
 *
 * No D1 / KV / Supabase — text is stored in Vectorize metadata.
 */

const SIMILARITY_THRESHOLD = 0.5
const TOP_K = 5

export default {
  async fetch(request, env) {
    const allowedOrigin = env.ALLOWED_ORIGIN || '*'

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders(allowedOrigin) })
    }

    const url = new URL(request.url)
    if (request.method !== 'POST' || url.pathname !== '/chat') {
      return new Response('Not found', { status: 404 })
    }

    try {
      const { message, history = [] } = await request.json()
      if (!message?.trim()) return jsonError('Message is required.', 400, allowedOrigin)

      // 1. Embed the user query via Workers AI (free, runs on CF edge)
      const queryEmbedding = await getEmbedding(env, message)

      // 2. Similarity search — metadata.text comes back directly, no extra DB call
      const context = await searchVectorize(env, queryEmbedding)

      // 3. Stream response from Gemini 2.0 Flash (free tier)
      const stream = await callGemini(env, message, context, history)

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          ...corsHeaders(allowedOrigin),
        },
      })
    } catch (err) {
      console.error('Worker error:', err)
      return jsonError('Server error. Please try again later.', 500, allowedOrigin)
    }
  },
}

/** Embed via Workers AI binding — no external API call, pure CF edge */
async function getEmbedding(env, text) {
  const result = await env.AI.run('@cf/baai/bge-base-en-v1.5', { text: [text] })
  return result.data[0]  // number[]
}

/** Query Vectorize and extract text from metadata — no DB lookup needed */
async function searchVectorize(env, embedding) {
  const results = await env.VECTORIZE.query(embedding, {
    topK: TOP_K,
    returnMetadata: 'all',   // includes metadata.text
  })

  return results.matches
    .filter(m => m.score >= SIMILARITY_THRESHOLD)
    .map(m => m.metadata?.text || '')
    .filter(Boolean)
    .join('\n\n---\n\n')
}

/** Gemini 2.0 Flash — free tier, 1,500 req/day, SSE streaming */
async function callGemini(env, userMessage, context, history) {
  const systemPrompt = context
    ? `You are Dogbro LLC's AI business assistant — professional, concise, and helpful.

Answer the user's question using only the company information below.
If the answer is not covered, say so honestly and suggest they contact our team.

[COMPANY KNOWLEDGE]
${context}`
    : `You are Dogbro LLC's AI business assistant — professional, concise, and helpful.
Answer the user's question to the best of your ability.
For specific business details, recommend they contact our team directly.`

  const contents = [
    ...history.slice(-10).map(h => ({
      role: h.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: h.content }],
    })),
    { role: 'user', parts: [{ text: userMessage }] },
  ]

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key=${env.GEMINI_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: systemPrompt }] },
        contents,
        generationConfig: { maxOutputTokens: 1024, temperature: 0.35 },
      }),
    }
  )

  if (!res.ok) throw new Error(`Gemini API error: ${await res.text()}`)

  const { readable, writable } = new TransformStream()
  const writer = writable.getWriter()
  const encoder = new TextEncoder()

  ;(async () => {
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value).split('\n')) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (raw === '[DONE]') continue
          try {
            const json = JSON.parse(raw)
            const text = json.candidates?.[0]?.content?.parts?.[0]?.text || ''
            if (text) {
              await writer.write(encoder.encode(`data: ${JSON.stringify({ text })}\n\n`))
            }
            if (json.candidates?.[0]?.finishReason === 'STOP') {
              await writer.write(encoder.encode('data: [DONE]\n\n'))
            }
          } catch { /* skip malformed SSE lines */ }
        }
      }
    } finally {
      await writer.close()
    }
  })()

  return readable
}

function corsHeaders(origin) {
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  }
}

function jsonError(message, status, origin) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { 'Content-Type': 'application/json', ...corsHeaders(origin) },
  })
}
