import makeWASocket, {
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  downloadMediaMessage
} from "@whiskeysockets/baileys";
import qrcode from "qrcode-terminal";
import { pipeline } from "@xenova/transformers";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import fetch from "node-fetch";
import dotenv from "dotenv";
dotenv.config();
process.env.HF_TOKEN = process.env.HF_TOKEN || dotenv.config().parsed?.HF_TOKEN;

// --- CONFIG ---
const KNOWLEDGE_DIR = path.resolve("./knowledge");
const VECTOR_INDEX_PATH = path.resolve("./vector_index.json");
const IMAGE_DIR = path.resolve("./images");
const IMAGE_INDEX_PATH = path.resolve("./image_index.json");
const SIM_THRESHOLD = 0.75;
const TOP_K = 3;

// --- HELPERS ---
const sha1 = (text) => crypto.createHash("sha1").update(text).digest("hex");
const log = (...args) => console.log("🧠", ...args);

// --- EMBEDDINGS (texto) ---
let embeddingPipeline = null;
async function getTextPipeline() {
  if (!embeddingPipeline) {
    log("🔄 Carregando modelo de embeddings (MiniLM)...");
    embeddingPipeline = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return embeddingPipeline;
}
async function embedText(text) {
  const pipe = await getTextPipeline();
  const output = await pipe(text, { pooling: "mean", normalize: true });
  return Array.from(output.data || output);
}

// --- IMAGE CAPTIONING (BLIP-Small) ---
let captionPipeline = null;
async function getCaptionPipeline() {
  if (!captionPipeline) {
    log("🔄 Carregando modelo de descrição de imagem (BLIP-Small)...");
    captionPipeline = await pipeline("image-to-text", "Xenova/blip-small");
  }
  return captionPipeline;
}
async function describeImage(filePath) {
  try {
    const pipe = await getCaptionPipeline();
    const buffer = fs.readFileSync(filePath);
    const result = await pipe(buffer);
    return result.generated_text || "Imagem recebida (sem descrição).";
  } catch (err) {
    console.error("Erro ao descrever imagem:", err?.message || err);
    return "Imagem recebida (não consegui descrever).";
  }
}

// --- IMAGE EMBEDDINGS (CLIP) ---
let imageEmbeddingPipeline = null;
async function getImagePipeline() {
  if (!imageEmbeddingPipeline) {
    log("🔄 Carregando modelo CLIP para embeddings de imagem...");
    imageEmbeddingPipeline = await pipeline("feature-extraction", "Xenova/clip-vit-base-patch32");
  }
  return imageEmbeddingPipeline;
}
async function embedImage(filePath) {
  const pipe = await getImagePipeline();
  const buffer = fs.readFileSync(filePath);
  const output = await pipe(buffer, { pooling: "mean", normalize: true });
  return Array.from(output.data || output);
}

// --- VETOR UTIL ---
function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
function norm(a) { return Math.sqrt(dot(a, a)); }
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  const na = norm(a), nb = norm(b);
  return na && nb ? dot(a, b) / (na * nb) : 0;
}

// --- INDEX (texto) ---
function loadIndex() {
  try {
    return fs.existsSync(VECTOR_INDEX_PATH)
      ? JSON.parse(fs.readFileSync(VECTOR_INDEX_PATH, "utf-8"))
      : { docs: [] };
  } catch {
    return { docs: [] };
  }
}
function saveIndex(index) {
  fs.writeFileSync(VECTOR_INDEX_PATH, JSON.stringify(index, null, 2));
}
function chunkByQA(text) {
  return text.split(/\n{2,}/).map(c => c.trim()).filter(Boolean);
}
async function buildIndex() {
  const index = { docs: [] };
  if (!fs.existsSync(KNOWLEDGE_DIR)) fs.mkdirSync(KNOWLEDGE_DIR, { recursive: true });
  const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f => /\.(txt|md)$/i.test(f));
  for (const file of files) {
    const raw = fs.readFileSync(path.join(KNOWLEDGE_DIR, file), "utf-8");
    const chunks = chunkByQA(raw);
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = await embedText(chunk);
      index.docs.push({ id: `${file}#${i}`, file, text: chunk, embedding });
    }
  }
  saveIndex(index);
  log("✅ Index criado com", index.docs.length, "chunks.");
  return index;
}

// --- IMAGE INDEX ---
function loadImageIndex() {
  try {
    if (!fs.existsSync(IMAGE_INDEX_PATH)) return [];
    return JSON.parse(fs.readFileSync(IMAGE_INDEX_PATH, "utf-8"));
  } catch {
    return [];
  }
}
function saveImageIndex(index) {
  fs.writeFileSync(IMAGE_INDEX_PATH, JSON.stringify(index, null, 2));
}
async function searchImageByText(query) {
  const qEmb = await embedText(query);
  const index = loadImageIndex();
  if (!index.length) return null;
  let best = null;
  let bestScore = -1;
  for (const img of index) {
    if (!img.embedding) continue;
    const score = cosineSim(qEmb, img.embedding);
    if (score > bestScore) {
      best = img;
      bestScore = score;
    }
  }
  return best;
}

// --- LLaMA LOCAL (Ollama) ---
async function generateWithLlama(prompt, opts = {}) {
  const body = {
    model: "llama2",
    prompt,
    stream: false,
    temperature: opts.temperature ?? 0.0,
    max_tokens: opts.max_tokens ?? 256
  };
  try {
    log("➡️ Enviando prompt ao LLaMA (preview):", prompt.slice(0, 200).replace(/\n/g, " "));
    const response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await response.json();
    return data.response || data?.text || "⚠️ Não consegui gerar resposta.";
  } catch (err) {
    console.error("Erro LLaMA:", err?.message || err);
    return "⚠️ LLaMA não disponível.";
  }
}

// --- RAG (com filtro para saudações) ---
async function ragRespond(query, index) {
  const lower = query.toLowerCase().trim();

  // Lista de saudações comuns
  const greetings = ["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "tudo bem", "eai", "e aí"];

  // Se for uma saudação simples -> responde direto no LLaMA (sem RAG)
  if (greetings.some(g => lower.startsWith(g))) {
    const prompt = `Responda de forma curta, simpática e em português, como se fosse uma conversa natural.

Mensagem: ${query}
Resposta:`;
    return await generateWithLlama(prompt, { temperature: 0.7, max_tokens: 50 });
  }

  // Caso contrário, segue a lógica do RAG
  const qEmb = await embedText(query);
  const results = index.docs
    .map(d => ({ ...d, score: cosineSim(qEmb, d.embedding || []) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_K);

  if (!results.length || results[0].score < SIM_THRESHOLD) {
    // Sem contexto suficiente -> apenas pergunta direto ao LLaMA
    const prompt = `Responda em português de forma clara e objetiva à pergunta abaixo.

Pergunta:
${query}

Resposta:`;
    return await generateWithLlama(prompt, { temperature: 0.0, max_tokens: 220 });
  }

  const context = results.map(r => `- ${r.text}`).join("\n");
  const prompt = `Você é um assistente técnico. Use o contexto abaixo para responder em português, de forma clara e objetiva.

Contexto:
${context}

Pergunta:
${query}

Resposta:`;

  return await generateWithLlama(prompt, { temperature: 0.0, max_tokens: 220 });
}

// --- BOT ---
async function startBot() {
  if (!fs.existsSync(IMAGE_DIR)) fs.mkdirSync(IMAGE_DIR, { recursive: true });

  let index = loadIndex();
  if (!index.docs || index.docs.length === 0) {
    log("📂 Index vazio. Construindo...");
    index = await buildIndex();
  }

  const { state, saveCreds } = await useMultiFileAuthState(path.resolve("./auth"));
  const { version } = await fetchLatestBaileysVersion();
  const sock = makeWASocket({ version, auth: state });

  sock.ev.on("creds.update", saveCreds);
  sock.ev.on("connection.update", ({ connection, qr, lastDisconnect }) => {
    if (qr) qrcode.generate(qr, { small: true });
    if (connection === "open") log("🟢 Conexão aberta");
    if (connection === "close") {
      const reason = lastDisconnect?.error?.output?.statusCode || "unknown";
      log("🔴 Conexão fechada, código:", reason, "Reconectando...");
      setTimeout(() => startBot(), 3000);
    }
  });

  sock.ev.on("messages.upsert", async (m) => {
    for (const msg of m.messages) {
      if (!msg.message || msg.key?.fromMe) continue;
      const jid = msg.key.remoteJid;
      const text =
        msg.message.conversation ||
        msg.message.extendedTextMessage?.text ||
        msg.message.imageMessage?.caption ||
        msg.message.videoMessage?.caption ||
        "";

      // --- imagem recebida ---
      if (msg.message.imageMessage) {
        try {
          const buffer = await downloadMediaMessage(msg, "buffer");
          const fileName = `${IMAGE_DIR}/${Date.now()}_${sha1(buffer.slice(0, 16))}.jpg`;
          fs.writeFileSync(fileName, buffer);

          const caption = await describeImage(fileName);
          const embedding = await embedImage(fileName);
          const imageIndex = loadImageIndex();
          imageIndex.push({ file: fileName, caption, embedding });
          saveImageIndex(imageIndex);

          await sock.sendMessage(jid, { text: `Imagem recebida e descrita como: "${caption}"` });
        } catch (err) {
          console.error("Erro processando imagem:", err);
          await sock.sendMessage(jid, { text: "Não consegui processar a imagem." });
        }
        continue;
      }

      // --- busca por imagem via texto ---
      if (/me manda uma foto de|foto de/i.test(text) && text.trim().length > 5) {
        const termo = text.replace(/me manda uma foto de/i, "").replace(/foto de/i, "").trim();
        const found = await searchImageByText(termo);
        if (found) {
          try {
            const data = fs.readFileSync(found.file);
            await sock.sendMessage(jid, { image: data, caption: found.caption });
          } catch (err) {
            console.error("Erro enviando imagem:", err);
            await sock.sendMessage(jid, { text: "Encontrei a imagem, mas não consegui enviá-la." });
          }
        } else {
          await sock.sendMessage(jid, { text: "Não encontrei nenhuma imagem correspondente." });
        }
        continue;
      }

      // --- texto / RAG ---
      if (text && text.trim()) {
        log(`📨 Mensagem de ${jid}:`, text);
        try {
          const reply = await ragRespond(text, index);
          await sock.sendMessage(jid, { text: reply });
        } catch (err) {
          console.error("Erro na resposta RAG:", err);
          await sock.sendMessage(jid, { text: "Erro ao processar sua solicitação." });
        }
      }
    }
  });
}

startBot().catch(console.error);
