//! Cloud speech-to-text backends.
//!
//! Handy transcribes locally by default (whisper.cpp / Parakeet). This module
//! adds an optional cloud path that bypasses the local engine entirely: the
//! recorded samples are shipped to a hosted STT API and the returned text
//! re-enters the normal pipeline (custom-word correction, then LLM
//! post-processing), so everything downstream of `transcribe()` is unchanged.
//!
//! Only Deepgram is implemented today, but the request struct is deliberately
//! provider-shaped so a second backend slots in beside it.

use anyhow::{anyhow, Result};
use log::{debug, warn};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use std::time::Duration;

pub const DEEPGRAM_PROVIDER_ID: &str = "deepgram";
pub const DEEPGRAM_DEFAULT_MODEL: &str = "nova-3";
/// Whether the model takes decode-time keyterm bias.
///
/// Callers use this to decide whether the local fuzzy custom-word pass still
/// needs to run: when Deepgram has already been told the terms, re-correcting
/// its output only risks rewriting words it got right.
pub fn model_accepts_keyterms(model: &str) -> bool {
    // keyterm is a nova-3 feature on /v1; older models 400 on it. Flux takes it
    // on /v2 regardless of variant.
    is_flux_model(model) || model.starts_with("nova-3")
}

/// Deepgram's conversational turn-based model. Streaming-only, and a different
/// wire protocol from the `/v1/listen` models — see [`build_flux_ws_url`].
///
/// The multilingual variant is the default because Handy's own language setting
/// defaults to "auto": `flux-general-multi` covers 10 languages (English among
/// them) and auto-detects, whereas `flux-general-en` would silently force
/// English on everyone. Users who only ever dictate English can switch to
/// `flux-general-en` for a marginal accuracy gain.
pub const DEEPGRAM_FLUX_MODEL: &str = "flux-general-multi";

/// Flux speaks `/v2/listen` with `TurnInfo` frames; everything else speaks
/// `/v1/listen` with `Results` frames. The model id is the only discriminator
/// Deepgram gives us.
fn is_flux_model(model: &str) -> bool {
    model.trim().starts_with("flux")
}

const DEEPGRAM_URL: &str = "https://api.deepgram.com/v1/listen";
/// Handy hands us mono f32 at the whisper rate; Deepgram is told the same.
const SAMPLE_RATE: u32 = crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Deepgram caps keyterm prompting; stay well under it rather than 400ing.
const MAX_KEYTERMS: usize = 100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudSttRequest {
    pub api_key: String,
    /// Deepgram model id, e.g. `nova-3`.
    pub model: String,
    /// Handy's language *intent* ("auto", "en", "fr", ...). "auto" becomes
    /// Deepgram's `multi`.
    pub language: String,
    /// Custom words, passed as `keyterm` so the model biases toward them at
    /// decode time (nova-3 only). The local fuzzy correction pass still runs
    /// afterwards, so this is bias, not a replacement.
    pub keyterms: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DeepgramResponse {
    results: Option<DeepgramResults>,
}

#[derive(Debug, Deserialize)]
struct DeepgramResults {
    channels: Vec<DeepgramChannel>,
}

#[derive(Debug, Deserialize)]
struct DeepgramChannel {
    alternatives: Vec<DeepgramAlternative>,
}

#[derive(Debug, Deserialize)]
struct DeepgramAlternative {
    transcript: Option<String>,
}

/// Convert Handy's mono f32 samples to little-endian 16-bit PCM, matching the
/// `encoding=linear16` we declare in the query string.
fn to_linear16(audio: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(audio.len() * 2);
    for sample in audio {
        let clamped = sample.clamp(-1.0, 1.0);
        let value = (clamped * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Deepgram has no "auto" — `multi` is its multilingual mode.
fn deepgram_language(language: &str) -> String {
    match language.trim() {
        "" | "auto" => "multi".to_string(),
        other => other.to_string(),
    }
}

/// The model to use on the batch endpoint. Flux is streaming-only, so a Flux
/// selection falls back to the default batch model rather than 400ing — the
/// batch path exists precisely as the safety net when the socket fails.
pub fn batch_model(model: &str) -> &str {
    if is_flux_model(model) {
        warn!(
            "{} is streaming-only; falling back to {} for the batch request",
            model, DEEPGRAM_DEFAULT_MODEL
        );
        DEEPGRAM_DEFAULT_MODEL
    } else {
        model
    }
}

fn build_url(req: &CloudSttRequest) -> Result<String> {
    let model = batch_model(if req.model.trim().is_empty() {
        DEEPGRAM_DEFAULT_MODEL
    } else {
        req.model.trim()
    });

    let mut url = reqwest::Url::parse(DEEPGRAM_URL)?;
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("model", model);
        q.append_pair("encoding", "linear16");
        q.append_pair("sample_rate", &SAMPLE_RATE.to_string());
        q.append_pair("channels", "1");
        q.append_pair("language", &deepgram_language(&req.language));
        q.append_pair("smart_format", "true");
        q.append_pair("punctuate", "true");

        if model_accepts_keyterms(model) {
            for term in req
                .keyterms
                .iter()
                .map(|t| t.trim())
                .filter(|t| !t.is_empty())
                .take(MAX_KEYTERMS)
            {
                q.append_pair("keyterm", term);
            }
        }
    }
    Ok(url.to_string())
}

fn build_headers(api_key: &str) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Token {}", api_key))
            .map_err(|_| anyhow!("Deepgram API key contains invalid characters"))?,
    );
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    Ok(headers)
}

pub async fn transcribe_deepgram(req: &CloudSttRequest, audio: &[f32]) -> Result<String> {
    if req.api_key.trim().is_empty() {
        return Err(anyhow!(
            "Deepgram API key is not set. Add it in Settings → Transcription."
        ));
    }

    let url = build_url(req)?;
    let body = to_linear16(audio);
    debug!(
        "Deepgram request: {} bytes of linear16 ({:.2}s of audio)",
        body.len(),
        audio.len() as f32 / SAMPLE_RATE as f32
    );

    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .default_headers(build_headers(&req.api_key)?)
        .build()?;

    let response = client
        .post(url)
        .body(body)
        .send()
        .await
        .map_err(|e| anyhow!("Deepgram request failed: {}", e))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|e| anyhow!("Failed to read Deepgram response: {}", e))?;

    if !status.is_success() {
        // The key itself is never echoed back by Deepgram, so the body is safe
        // to surface — it carries the actual reason (bad model, bad key, ...).
        return Err(anyhow!("Deepgram returned {}: {}", status, text.trim()));
    }

    parse_transcript(&text)
}

/// Blocking wrapper for the synchronous `TranscriptionManager::transcribe`.
///
/// `transcribe()` is sync, but its main caller (`actions.rs`) invokes it from
/// inside `tauri::async_runtime::spawn(async move { .. })` — i.e. from a tokio
/// worker thread. Calling `block_on` on that same runtime deadlocks, so the
/// request gets its own OS thread with a private current-thread runtime. The
/// caller still blocks, which is what the sync signature promises and what the
/// local engine does anyway.
pub fn transcribe_deepgram_blocking(req: &CloudSttRequest, audio: &[f32]) -> Result<String> {
    let req = req.clone();
    let audio = audio.to_vec();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| anyhow!("Failed to start HTTP runtime: {}", e))?;
        rt.block_on(transcribe_deepgram(&req, &audio))
    })
    .join()
    .map_err(|_| anyhow!("Deepgram request thread panicked"))?
}

/* ── Streaming (Deepgram live WebSocket) ─────────────────────────────────── */

const DEEPGRAM_WS_URL: &str = "wss://api.deepgram.com/v1/listen";
/// How long to wait for the WebSocket handshake before giving up and letting
/// the caller fall back to the batch endpoint.
const WS_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long to wait for Deepgram to flush its finals after `CloseStream`.
/// Measured round-trip is ~1.5s; this is a generous ceiling that still leaves
/// headroom under the caller's own 30s finalize timeout.
const WS_FINALIZE_TIMEOUT: Duration = Duration::from_secs(12);

/// Messages pushed to the socket-owning task.
enum WsCmd {
    Audio(Vec<u8>),
    /// Ask Deepgram to flush; the task replies on the result channel.
    Close,
}

/// A live Deepgram transcription session.
///
/// The batch endpoint (`transcribe_deepgram`) only starts working once the
/// recording is over, and measured 4–8s of server-side latency before the first
/// byte — the whole reason this exists. Here the socket is opened at
/// record-start and fed as the user speaks, so by the time they release the key
/// Deepgram has already transcribed everything but the last moment of audio.
///
/// Owns an OS thread with a private current-thread runtime, for the same reason
/// [`transcribe_deepgram_blocking`] does: the callers are synchronous and may
/// themselves be running on a Tauri runtime worker.
pub struct DeepgramLiveStream {
    cmd_tx: tokio::sync::mpsc::UnboundedSender<WsCmd>,
    /// Final transcript (or the error that killed the session), sent exactly
    /// once when the socket task finishes.
    result_rx: std::sync::mpsc::Receiver<Result<String>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl DeepgramLiveStream {
    /// Whether the socket task is still running.
    ///
    /// The task owns the command receiver, so the sender closing means the task
    /// exited -- Deepgram dropped an idle connection, or the session errored.
    /// A pre-warmed stream must be checked with this before use: reusing a dead
    /// socket would fail the dictation into the slow batch fallback, which is
    /// worse than simply paying for a fresh connect.
    pub fn is_alive(&self) -> bool {
        !self.cmd_tx.is_closed()
    }


    /// Open the socket. Blocks until the handshake completes (or fails), so a
    /// returned stream is ready to accept audio.
    ///
    /// `on_interim` is invoked from the socket task for every result Deepgram
    /// sends, with `(committed, tentative)` — the finalized prefix so far and
    /// the volatile tail — matching the shape the streaming overlay expects.
    pub fn connect(
        req: &CloudSttRequest,
        on_interim: impl Fn(&str, &str) + Send + 'static,
    ) -> Result<Self> {
        if req.api_key.trim().is_empty() {
            return Err(anyhow!(
                "Deepgram API key is not set. Add it in Settings → Transcription."
            ));
        }

        let url = build_ws_url(req)?;
        let flux = is_flux_model(if req.model.trim().is_empty() {
            DEEPGRAM_DEFAULT_MODEL
        } else {
            req.model.trim()
        });
        let api_key = req.api_key.clone();
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel::<WsCmd>();
        // Two hops: `ready` reports handshake success so `connect` can block,
        // `result` carries the final transcript.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<Result<String>>();

        let handle = std::thread::Builder::new()
            .name("deepgram-live".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx.send(Err(anyhow!("Failed to start WS runtime: {}", e)));
                        return;
                    }
                };
                rt.block_on(run_ws_session(
                    url,
                    api_key,
                    flux,
                    cmd_rx,
                    ready_tx,
                    result_tx,
                    on_interim,
                ));
            })
            .map_err(|e| anyhow!("Failed to spawn Deepgram stream thread: {}", e))?;

        match ready_rx.recv_timeout(WS_CONNECT_TIMEOUT) {
            Ok(Ok(())) => Ok(Self {
                cmd_tx,
                result_rx,
                handle: Some(handle),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(anyhow!(
                "Timed out after {:?} connecting to Deepgram",
                WS_CONNECT_TIMEOUT
            )),
        }
    }

    /// Push one frame of mono f32 audio. Non-blocking; a dead socket is
    /// silently dropped here and surfaced by [`finalize`](Self::finalize).
    pub fn feed(&self, pcm: &[f32]) {
        let _ = self.cmd_tx.send(WsCmd::Audio(to_linear16(pcm)));
    }

    /// Flush the stream and return the full transcript.
    pub fn finalize(mut self) -> Result<String> {
        let _ = self.cmd_tx.send(WsCmd::Close);
        let out = match self.result_rx.recv_timeout(WS_FINALIZE_TIMEOUT) {
            Ok(result) => result,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("Deepgram stream ended without a transcript"))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(anyhow!(
                "Timed out after {:?} waiting for Deepgram to finalize",
                WS_FINALIZE_TIMEOUT
            )),
        };
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        out
    }
}

impl Drop for DeepgramLiveStream {
    fn drop(&mut self) {
        // Cancelled mid-recording: dropping the sender ends the task's recv
        // loop, which closes the socket. The thread is detached rather than
        // joined so cancel stays instant.
        if let Some(handle) = self.handle.take() {
            drop(handle);
        }
    }
}

fn build_ws_url(req: &CloudSttRequest) -> Result<String> {
    let model = if req.model.trim().is_empty() {
        DEEPGRAM_DEFAULT_MODEL
    } else {
        req.model.trim()
    };

    if is_flux_model(model) {
        return build_flux_ws_url(req, model);
    }

    let mut url = reqwest::Url::parse(DEEPGRAM_WS_URL)?;
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("model", model);
        q.append_pair("encoding", "linear16");
        q.append_pair("sample_rate", &SAMPLE_RATE.to_string());
        q.append_pair("channels", "1");
        q.append_pair("language", &deepgram_language(&req.language));
        q.append_pair("smart_format", "true");
        q.append_pair("punctuate", "true");
        // Interim results drive the live overlay; without them nothing appears
        // until the user stops speaking.
        q.append_pair("interim_results", "true");

        if model_accepts_keyterms(model) {
            for term in req
                .keyterms
                .iter()
                .map(|t| t.trim())
                .filter(|t| !t.is_empty())
                .take(MAX_KEYTERMS)
            {
                q.append_pair("keyterm", term);
            }
        }
    }
    Ok(url.to_string())
}

const DEEPGRAM_FLUX_WS_URL: &str = "wss://api.deepgram.com/v2/listen";

/// Flux's query string. Deliberately not the `/v1` one: Flux punctuates on its
/// own, has no `interim_results` (it streams cumulative turns instead), and
/// takes end-of-turn thresholds in place of `endpointing`.
///
/// It does *not* format numbers on its own. `/v1`'s `smart_format=true` implied
/// numerals; Flux splits that out into `numerals`, which defaults to **false**.
/// Dropping `smart_format` without adding this is what turns "July 2024" back
/// into "July twenty twenty four".
///
/// Handy is push-to-talk, so the *user's key release* is the real end of the
/// utterance — not Deepgram's guess at one. Both thresholds are therefore
/// pushed to their maximums so Flux won't guillotine a turn just because the
/// speaker paused to think. We end the turn ourselves on `CloseStream`.
fn build_flux_ws_url(req: &CloudSttRequest, model: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(DEEPGRAM_FLUX_WS_URL)?;
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("model", model);
        q.append_pair("encoding", "linear16");
        q.append_pair("sample_rate", &SAMPLE_RATE.to_string());
        // Deepgram's defaults here are 0.7 and 5000; both are pushed to their
        // maximums on purpose (see above).
        q.append_pair("eot_threshold", "0.9");
        q.append_pair("eot_timeout_ms", "60000");
        // Spoken dates, times, quantities and money as digits. Defaults to
        // false on Flux, unlike `/v1` where `smart_format` covered it.
        q.append_pair("numerals", "true");

        // The multilingual variant takes hints; the English one takes no
        // language parameter at all and 400s if given one.
        if model.contains("multi") {
            let lang = deepgram_language(&req.language);
            if lang != "multi" {
                q.append_pair("language_hint", &lang);
            }
        }

        for term in req
            .keyterms
            .iter()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .take(MAX_KEYTERMS)
        {
            q.append_pair("keyterm", term);
        }
    }
    Ok(url.to_string())
}

/// One `TurnInfo` frame from Flux.
#[derive(Debug, Deserialize)]
struct FluxLiveMessage {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    /// `StartOfTurn` | `Update` | `EagerEndOfTurn` | `TurnResumed` | `EndOfTurn`
    event: Option<String>,
    turn_index: Option<u32>,
    transcript: Option<String>,
    /// Present on `type: "Error"` frames.
    code: Option<String>,
    description: Option<String>,
}

/// Accumulates Flux turns into a transcript.
///
/// Unlike `/v1`, a Flux transcript is *cumulative within a turn* and resets
/// when that turn ends — so the newest text for a given `turn_index` always
/// supersedes the previous one, and turns only ever accumulate forwards. That
/// maps cleanly onto the overlay's (committed, tentative) contract: every
/// closed turn is committed, and the still-open turn is the volatile tail.
#[derive(Default)]
struct FluxTranscript {
    turns: std::collections::BTreeMap<u32, String>,
    open: Option<u32>,
}

impl FluxTranscript {
    /// Returns true if anything changed and the caller should emit an update.
    fn apply(&mut self, index: u32, transcript: &str, end_of_turn: bool) -> bool {
        let text = transcript.trim();
        let mut changed = false;

        // An empty frame (a bare StartOfTurn, or silence closing a turn) must
        // never erase text we already hold for that turn.
        if !text.is_empty() && self.turns.get(&index).map(String::as_str) != Some(text) {
            self.turns.insert(index, text.to_string());
            changed = true;
        }

        let open = if end_of_turn { None } else { Some(index) };
        if self.open != open {
            self.open = open;
            changed = true;
        }
        changed
    }

    fn committed(&self) -> String {
        self.turns
            .iter()
            .filter(|(i, _)| Some(**i) != self.open)
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn tentative(&self) -> String {
        self.open
            .and_then(|i| self.turns.get(&i))
            .cloned()
            .unwrap_or_default()
    }

    fn display(&self) -> String {
        self.turns
            .values()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string()
    }
}

/// One `Results` frame from the live API.
#[derive(Debug, Deserialize)]
struct DeepgramLiveMessage {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    channel: Option<DeepgramChannel>,
    #[serde(default)]
    is_final: bool,
    /// Present on `type: "Error"` frames.
    description: Option<String>,
}

/// Accumulates Deepgram's interim/final result frames into a transcript.
///
/// The live API sends a stream of interim hypotheses for the current utterance
/// and then one `is_final` frame that supersedes them all. So finals append to
/// a committed prefix, and interims only ever replace the volatile tail.
#[derive(Default)]
struct LiveTranscript {
    committed: Vec<String>,
    tentative: String,
}

impl LiveTranscript {
    /// Returns true if anything changed and the caller should emit an update.
    fn apply(&mut self, transcript: &str, is_final: bool) -> bool {
        let text = transcript.trim();
        if is_final {
            // A final with no words is Deepgram closing out silence — it still
            // clears the tentative tail.
            let had_tentative = !self.tentative.is_empty();
            self.tentative.clear();
            if text.is_empty() {
                return had_tentative;
            }
            self.committed.push(text.to_string());
            true
        } else {
            if self.tentative == text {
                return false;
            }
            self.tentative = text.to_string();
            true
        }
    }

    fn committed(&self) -> String {
        self.committed.join(" ")
    }

    fn display(&self) -> String {
        let mut out = self.committed();
        if !self.tentative.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&self.tentative);
        }
        out.trim().to_string()
    }
}

async fn run_ws_session(
    url: String,
    api_key: String,
    // True for `/v2/listen` (Flux `TurnInfo` frames), false for `/v1/listen`.
    flux: bool,
    mut cmd_rx: tokio::sync::mpsc::UnboundedReceiver<WsCmd>,
    ready_tx: std::sync::mpsc::Sender<Result<()>>,
    result_tx: std::sync::mpsc::Sender<Result<String>>,
    on_interim: impl Fn(&str, &str),
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, Message};

    let socket = async {
        let mut request = url
            .into_client_request()
            .map_err(|e| anyhow!("Invalid Deepgram URL: {}", e))?;
        request.headers_mut().insert(
            "Authorization",
            format!("Token {}", api_key)
                .parse()
                .map_err(|_| anyhow!("Deepgram API key contains invalid characters"))?,
        );
        tokio_tungstenite::connect_async(request)
            .await
            .map(|(socket, _)| socket)
            .map_err(|e| anyhow!("Deepgram WebSocket connect failed: {}", e))
    }
    .await;

    let mut socket = match socket {
        Ok(s) => {
            let _ = ready_tx.send(Ok(()));
            s
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    debug!("Deepgram live stream connected");

    let mut transcript = LiveTranscript::default();
    let mut flux_transcript = FluxTranscript::default();
    // Set once `CloseStream` is sent: from then on we only drain replies.
    let mut closing = false;
    // Flux only: audio has been streamed that no `EndOfTurn` has covered yet, so
    // there is still a tail for the server to transcribe.
    let mut pending_audio = false;
    let mut outcome: Option<Result<String>> = None;

    loop {
        tokio::select! {
            // Bias toward draining the socket so interim results stay timely
            // even while frames are arriving.
            biased;

            incoming = socket.next() => {
                match incoming {
                    Some(Ok(Message::Text(payload))) if flux => {
                        match serde_json::from_str::<FluxLiveMessage>(&payload) {
                            Ok(msg) => {
                                match msg.msg_type.as_deref() {
                                    Some("Error") => {
                                        outcome = Some(Err(anyhow!(
                                            "Deepgram Flux error: {}",
                                            msg.description
                                                .or(msg.code)
                                                .unwrap_or_else(|| payload.to_string())
                                        )));
                                        break;
                                    }
                                    Some("TurnInfo") => {
                                        let index = msg.turn_index.unwrap_or(0);
                                        let ended = msg.event.as_deref() == Some("EndOfTurn");
                                        if ended {
                                            pending_audio = false;
                                        }
                                        let text = msg.transcript.unwrap_or_default();
                                        if flux_transcript.apply(index, &text, ended) {
                                            on_interim(
                                                &flux_transcript.committed(),
                                                &flux_transcript.tentative(),
                                            );
                                        }
                                        // The EndOfTurn that arrives after CloseStream covers
                                        // our trailing audio — that, not a timer, is when the
                                        // transcript is actually complete.
                                        if closing && ended {
                                            break;
                                        }
                                    }
                                    // Connected / ConfigureSuccess / keepalives.
                                    _ => {}
                                }
                            }
                            Err(e) => warn!("Unparseable Deepgram Flux frame: {} ({})", e, payload),
                        }
                    }
                    Some(Ok(Message::Text(payload))) => {
                        match serde_json::from_str::<DeepgramLiveMessage>(&payload) {
                            Ok(msg) => {
                                match msg.msg_type.as_deref() {
                                    Some("Error") => {
                                        outcome = Some(Err(anyhow!(
                                            "Deepgram stream error: {}",
                                            msg.description.unwrap_or_else(|| payload.to_string())
                                        )));
                                        break;
                                    }
                                    // Metadata is the last frame after CloseStream.
                                    Some("Metadata") => break,
                                    _ => {
                                        let text = msg
                                            .channel
                                            .and_then(|c| c.alternatives.into_iter().next())
                                            .and_then(|a| a.transcript)
                                            .unwrap_or_default();
                                        if transcript.apply(&text, msg.is_final) {
                                            on_interim(&transcript.committed(), &transcript.tentative);
                                        }
                                    }
                                }
                            }
                            Err(e) => warn!("Unparseable Deepgram frame: {} ({})", e, payload),
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        outcome = Some(Err(anyhow!("Deepgram WebSocket error: {}", e)));
                        break;
                    }
                }
            }

            cmd = cmd_rx.recv(), if !closing => {
                match cmd {
                    Some(WsCmd::Audio(bytes)) => {
                        pending_audio = true;
                        if let Err(e) = socket.send(Message::Binary(bytes.into())).await {
                            outcome = Some(Err(anyhow!("Failed to send audio to Deepgram: {}", e)));
                            break;
                        }
                    }
                    Some(WsCmd::Close) => {
                        closing = true;
                        let close_frame = r#"{"type":"CloseStream"}"#.to_string();
                        if let Err(e) = socket.send(Message::Text(close_frame.into())).await {
                            outcome = Some(Err(anyhow!("Failed to close Deepgram stream: {}", e)));
                            break;
                        }
                        // Flux: if every byte we sent is already covered by an EndOfTurn,
                        // the transcript is complete and no further frame is coming — don't
                        // sit here until the finalize timeout waiting for one.
                        if flux && !pending_audio {
                            break;
                        }
                    }
                    // Sender dropped without finalizing (cancelled recording).
                    None => break,
                }
            }
        }
    }

    let _ = socket.close(None).await;
    // An empty transcript is a legitimate result (silence), not an error —
    // same contract as the batch path.
    let final_text = if flux {
        flux_transcript.display()
    } else {
        transcript.display()
    };
    let _ = result_tx.send(outcome.unwrap_or_else(|| Ok(final_text)));
}

fn parse_transcript(body: &str) -> Result<String> {
    let parsed: DeepgramResponse = serde_json::from_str(body)
        .map_err(|e| anyhow!("Could not parse Deepgram response: {} (body: {})", e, body))?;

    // An empty/silent recording is a successful response with an empty
    // transcript, not an error — mirror the local engine and return "".
    let transcript = parsed
        .results
        .and_then(|r| r.channels.into_iter().next())
        .and_then(|c| c.alternatives.into_iter().next())
        .and_then(|a| a.transcript)
        .unwrap_or_default();

    Ok(transcript.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> CloudSttRequest {
        CloudSttRequest {
            api_key: "k".to_string(),
            model: "nova-3".to_string(),
            language: "auto".to_string(),
            keyterms: vec![],
        }
    }

    #[test]
    fn linear16_is_little_endian_i16() {
        // +1.0 saturates to i16::MAX, -1.0 to -i16::MAX, 0.0 to 0.
        let bytes = to_linear16(&[0.0, 1.0, -1.0]);
        assert_eq!(bytes.len(), 6);
        assert_eq!(&bytes[0..2], &0i16.to_le_bytes());
        assert_eq!(&bytes[2..4], &i16::MAX.to_le_bytes());
        assert_eq!(&bytes[4..6], &(-i16::MAX).to_le_bytes());
    }

    #[test]
    fn linear16_clamps_out_of_range_samples() {
        // Without the clamp, 2.0 * i16::MAX wraps to a negative sample.
        let bytes = to_linear16(&[2.0, -2.0]);
        assert_eq!(&bytes[0..2], &i16::MAX.to_le_bytes());
        assert_eq!(&bytes[2..4], &(-i16::MAX).to_le_bytes());
    }

    /// Pinned to the English variant on purpose: these assertions are about the
    /// `/v2` wire format, not about which model happens to be the default.
    fn flux_req() -> CloudSttRequest {
        CloudSttRequest {
            model: "flux-general-en".to_string(),
            ..req()
        }
    }

    #[test]
    fn only_keyterm_capable_models_suppress_the_fuzzy_pass() {
        // Both Flux variants and nova-3 take the terms at decode time, so the
        // caller skips the local Soundex correction for them.
        assert!(model_accepts_keyterms("flux-general-multi"));
        assert!(model_accepts_keyterms("flux-general-en"));
        assert!(model_accepts_keyterms("nova-3"));
        assert!(model_accepts_keyterms("nova-3-general"));
        // nova-2 and friends 400 on keyterm, so they never get one -- the fuzzy
        // pass is the only thing applying custom words there and must stay on.
        assert!(!model_accepts_keyterms("nova-2"));
        assert!(!model_accepts_keyterms("whisper-large"));
    }

    #[test]
    fn flux_still_counts_as_prompted_after_the_batch_fallback() {
        // Flux falls back to nova-3 off the socket, which also takes keyterms --
        // so the fallback must not silently re-enable the fuzzy pass.
        assert!(model_accepts_keyterms(batch_model("flux-general-multi")));
    }

    #[test]
    fn flux_asks_for_numerals() {
        // Flux defaults numerals to false, and it has no `smart_format` to imply
        // it -- without this, "twenty twenty four" never becomes "2024".
        let url = build_ws_url(&flux_req()).unwrap();
        assert!(url.contains("numerals=true"), "url was {url}");
        // Still no /v1-only formatting params, which Flux rejects.
        assert!(!url.contains("smart_format"));
    }

    #[test]
    fn the_default_flux_model_is_multilingual() {
        // Handy's language setting defaults to "auto", so the default Flux model
        // must not silently force English.
        assert!(is_flux_model(DEEPGRAM_FLUX_MODEL));
        assert!(DEEPGRAM_FLUX_MODEL.contains("multi"));
    }

    #[test]
    fn flux_models_are_detected_by_prefix() {
        assert!(is_flux_model("flux-general-en"));
        assert!(is_flux_model("flux-general-multi"));
        assert!(is_flux_model("  flux-general-en  "));
        assert!(!is_flux_model("nova-3"));
        assert!(!is_flux_model("nova-3-flux"));
    }

    #[test]
    fn flux_uses_the_v2_endpoint_and_turn_thresholds() {
        let url = build_ws_url(&flux_req()).unwrap();
        assert!(url.starts_with("wss://api.deepgram.com/v2/listen"));
        assert!(url.contains("model=flux-general-en"));
        assert!(url.contains("encoding=linear16"));
        assert!(url.contains("sample_rate=16000"));
        // Push-to-talk: we decide when the turn ends, not Deepgram.
        assert!(url.contains("eot_threshold=0.9"));
        assert!(url.contains("eot_timeout_ms=60000"));
        // /v1-only parameters must not leak onto /v2.
        assert!(!url.contains("interim_results"));
        assert!(!url.contains("smart_format"));
        assert!(!url.contains("punctuate"));
        // flux-general-en takes no language parameter at all.
        assert!(!url.contains("language"));
    }

    #[test]
    fn nova_still_uses_the_v1_endpoint() {
        let url = build_ws_url(&req()).unwrap();
        assert!(url.starts_with("wss://api.deepgram.com/v1/listen"));
        assert!(url.contains("interim_results=true"));
    }

    #[test]
    fn flux_multi_passes_a_language_hint_but_not_for_auto() {
        let hinted = build_ws_url(&CloudSttRequest {
            model: "flux-general-multi".to_string(),
            language: "fr".to_string(),
            ..req()
        })
        .unwrap();
        assert!(hinted.contains("language_hint=fr"));

        // "auto" maps to Deepgram's "multi", which is the model's own default —
        // sending it as a hint would pointlessly narrow nothing.
        let auto = build_ws_url(&CloudSttRequest {
            model: "flux-general-multi".to_string(),
            ..req()
        })
        .unwrap();
        assert!(!auto.contains("language_hint"));
    }

    #[test]
    fn flux_forwards_keyterms() {
        let url = build_ws_url(&CloudSttRequest {
            keyterms: vec!["Handy".into(), "  ".into(), "Deepgram".into()],
            ..flux_req()
        })
        .unwrap();
        assert!(url.contains("keyterm=Handy"));
        assert!(url.contains("keyterm=Deepgram"));
        // blank entries are dropped rather than sent as empty pairs
        assert_eq!(url.matches("keyterm=").count(), 2);
    }

    #[test]
    fn batch_falls_back_off_flux() {
        // Flux is streaming-only; the batch safety net must not 400.
        assert_eq!(batch_model("flux-general-en"), DEEPGRAM_DEFAULT_MODEL);
        assert_eq!(batch_model("nova-3"), "nova-3");
        let url = build_url(&flux_req()).unwrap();
        assert!(url.contains("model=nova-3"));
    }

    #[test]
    fn flux_transcript_supersedes_within_a_turn() {
        let mut t = FluxTranscript::default();
        // Cumulative: the newest text for a turn replaces the older one.
        assert!(t.apply(0, "hello", false));
        assert!(t.apply(0, "hello there", false));
        assert_eq!(t.tentative(), "hello there");
        assert_eq!(t.committed(), "");
        assert_eq!(t.display(), "hello there");
        // Identical repeat is not a change.
        assert!(!t.apply(0, "hello there", false));
    }

    #[test]
    fn flux_transcript_commits_on_end_of_turn() {
        let mut t = FluxTranscript::default();
        t.apply(0, "first turn", false);
        assert!(t.apply(0, "first turn", true));
        assert_eq!(t.committed(), "first turn");
        assert_eq!(t.tentative(), "");

        t.apply(1, "second", false);
        assert_eq!(t.committed(), "first turn");
        assert_eq!(t.tentative(), "second");
        assert_eq!(t.display(), "first turn second");
    }

    #[test]
    fn flux_transcript_ignores_empty_frames() {
        let mut t = FluxTranscript::default();
        t.apply(0, "kept", false);
        // A bare StartOfTurn / silence frame must not erase what we hold.
        t.apply(0, "", false);
        assert_eq!(t.display(), "kept");
        t.apply(0, "   ", true);
        assert_eq!(t.committed(), "kept");
    }

    #[test]
    fn flux_turns_are_ordered_by_index_not_arrival() {
        let mut t = FluxTranscript::default();
        t.apply(2, "third", true);
        t.apply(0, "first", true);
        t.apply(1, "second", true);
        assert_eq!(t.display(), "first second third");
    }

    #[test]
    fn flux_error_frame_parses() {
        let msg: FluxLiveMessage = serde_json::from_str(
            r#"{"type":"Error","code":"INVALID_AUTH","description":"bad key"}"#,
        )
        .unwrap();
        assert_eq!(msg.msg_type.as_deref(), Some("Error"));
        assert_eq!(msg.description.as_deref(), Some("bad key"));
    }

    #[test]
    fn flux_turninfo_frame_parses() {
        let msg: FluxLiveMessage = serde_json::from_str(
            r#"{"type":"TurnInfo","event":"EndOfTurn","turn_index":3,
                "transcript":"all done","end_of_turn_confidence":0.94,"words":[]}"#,
        )
        .unwrap();
        assert_eq!(msg.event.as_deref(), Some("EndOfTurn"));
        assert_eq!(msg.turn_index, Some(3));
        assert_eq!(msg.transcript.as_deref(), Some("all done"));
    }

    #[test]
    fn auto_language_maps_to_multi() {
        assert_eq!(deepgram_language("auto"), "multi");
        assert_eq!(deepgram_language(""), "multi");
        assert_eq!(deepgram_language("en"), "en");
    }

    #[test]
    fn url_carries_pcm_params_and_keyterms() {
        let mut r = req();
        r.keyterms = vec!["Safzan".into(), "  ".into(), "cliproxy".into()];
        let url = build_url(&r).unwrap();
        assert!(url.contains("model=nova-3"));
        assert!(url.contains("encoding=linear16"));
        assert!(url.contains("sample_rate=16000"));
        assert!(url.contains("channels=1"));
        assert!(url.contains("language=multi"));
        assert!(url.contains("keyterm=Safzan"));
        assert!(url.contains("keyterm=cliproxy"));
        // Blank entries are dropped rather than sent as empty keyterms.
        assert_eq!(url.matches("keyterm=").count(), 2);
    }

    #[test]
    fn keyterms_are_omitted_for_non_nova3_models() {
        let mut r = req();
        r.model = "base-general".into();
        r.keyterms = vec!["Safzan".into()];
        let url = build_url(&r).unwrap();
        assert!(!url.contains("keyterm"));
        assert!(url.contains("model=base-general"));
    }

    #[test]
    fn empty_model_falls_back_to_default() {
        let mut r = req();
        r.model = "   ".into();
        assert!(build_url(&r).unwrap().contains("model=nova-3"));
    }

    #[test]
    fn parses_transcript_out_of_response() {
        let body = r#"{"results":{"channels":[{"alternatives":[{"transcript":" hello world "}]}]}}"#;
        assert_eq!(parse_transcript(body).unwrap(), "hello world");
    }

    #[test]
    fn silent_audio_yields_empty_string_not_error() {
        let body = r#"{"results":{"channels":[{"alternatives":[{"transcript":""}]}]}}"#;
        assert_eq!(parse_transcript(body).unwrap(), "");
        let no_channels = r#"{"results":{"channels":[]}}"#;
        assert_eq!(parse_transcript(no_channels).unwrap(), "");
    }

    #[test]
    fn ws_url_is_wss_and_requests_interim_results() {
        let url = build_ws_url(&req()).unwrap();
        assert!(url.starts_with("wss://api.deepgram.com/v1/listen"));
        assert!(url.contains("interim_results=true"));
        assert!(url.contains("encoding=linear16"));
        assert!(url.contains("sample_rate=16000"));
        assert!(url.contains("language=multi"));
    }

    #[test]
    fn interims_replace_the_tail_and_finals_append() {
        let mut t = LiveTranscript::default();

        assert!(t.apply("hello", false));
        assert_eq!(t.display(), "hello");
        // An identical interim is not a change — no redundant overlay emit.
        assert!(!t.apply("hello", false));

        // The interim is a hypothesis for the current utterance, so a longer
        // one replaces it rather than appending.
        assert!(t.apply("hello wor", false));
        assert_eq!(t.display(), "hello wor");

        // The final supersedes every interim for that utterance.
        assert!(t.apply("hello world", true));
        assert_eq!(t.display(), "hello world");
        assert_eq!(t.committed(), "hello world");
        assert_eq!(t.tentative, "");

        // The next utterance appends after the committed prefix.
        assert!(t.apply("how are you", false));
        assert_eq!(t.display(), "hello world how are you");
        assert!(t.apply("how are you?", true));
        assert_eq!(t.display(), "hello world how are you?");
    }

    #[test]
    fn empty_final_clears_a_stale_tentative_tail() {
        // Deepgram closes out a silence-only segment with an empty final; the
        // tail it supersedes must not survive into the transcript.
        let mut t = LiveTranscript::default();
        t.apply("uh", false);
        assert!(t.apply("", true));
        assert_eq!(t.display(), "");
        // ...and a second empty final is not a change worth emitting.
        assert!(!t.apply("", true));
    }

    #[test]
    fn silent_stream_finalizes_to_empty_string() {
        assert_eq!(LiveTranscript::default().display(), "");
    }

    #[test]
    fn live_results_frame_parses() {
        let frame = r#"{"type":"Results","channel":{"alternatives":[{"transcript":"hello"}]},"is_final":true}"#;
        let msg: DeepgramLiveMessage = serde_json::from_str(frame).unwrap();
        assert_eq!(msg.msg_type.as_deref(), Some("Results"));
        assert!(msg.is_final);
        assert_eq!(
            msg.channel.unwrap().alternatives[0].transcript.as_deref(),
            Some("hello")
        );
    }

    #[test]
    fn metadata_and_error_frames_parse_without_a_channel() {
        // Both are terminal frames; neither carries a transcript.
        let meta: DeepgramLiveMessage =
            serde_json::from_str(r#"{"type":"Metadata","duration":1.5}"#).unwrap();
        assert_eq!(meta.msg_type.as_deref(), Some("Metadata"));
        assert!(meta.channel.is_none());
        assert!(!meta.is_final);

        let err: DeepgramLiveMessage =
            serde_json::from_str(r#"{"type":"Error","description":"bad key"}"#).unwrap();
        assert_eq!(err.description.as_deref(), Some("bad key"));
    }

    #[test]
    fn malformed_body_is_an_error() {
        assert!(parse_transcript("not json").is_err());
    }

    /// Regression: `transcribe()` is sync but runs inside the Tauri async
    /// runtime, so an earlier version that called `block_on` on that runtime
    /// hung forever with the overlay stuck on "transcribing". The wrapper must
    /// return normally when called from inside a runtime. An empty API key
    /// makes this hermetic — it fails before any network call, while still
    /// exercising the thread + nested-runtime path that deadlocked.
    #[test]
    fn blocking_wrapper_returns_when_called_from_inside_a_runtime() {
        let mut r = req();
        r.api_key = String::new();

        let outer = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let result = outer.block_on(async { transcribe_deepgram_blocking(&r, &[0.0; 16]) });

        let err = result.expect_err("empty key should be rejected");
        assert!(
            err.to_string().contains("API key is not set"),
            "unexpected error: {err}"
        );
    }
}
