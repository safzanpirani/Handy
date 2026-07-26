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
use log::debug;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use std::time::Duration;

pub const DEEPGRAM_PROVIDER_ID: &str = "deepgram";
pub const DEEPGRAM_DEFAULT_MODEL: &str = "nova-3";

const DEEPGRAM_URL: &str = "https://api.deepgram.com/v1/listen";
/// Handy hands us mono f32 at the whisper rate; Deepgram is told the same.
const SAMPLE_RATE: u32 = crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Deepgram caps keyterm prompting; stay well under it rather than 400ing.
const MAX_KEYTERMS: usize = 100;

#[derive(Debug, Clone)]
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

fn build_url(req: &CloudSttRequest) -> Result<String> {
    let model = if req.model.trim().is_empty() {
        DEEPGRAM_DEFAULT_MODEL
    } else {
        req.model.trim()
    };

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

        // keyterm is a nova-3 feature; older models 400 on it.
        if model.starts_with("nova-3") {
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
