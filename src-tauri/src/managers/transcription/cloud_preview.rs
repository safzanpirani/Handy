//! Live preview for the custom cloud endpoint, which only transcribes whole
//! files: while recording, the audio so far is sent again at an interval and
//! each answer is shown as tentative overlay text.
//!
//! A preview never becomes the transcript. Finalize replies `None`, so the
//! recording is transcribed in one batch request at key release, exactly as
//! without previews.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, info, warn};
use tauri::async_runtime::JoinHandle;

use super::{
    cloud_request, drain_until_finalize, emit_stream_text, StreamCmd, StreamWorkerGuard,
    TranscriptionManager,
};
use crate::settings::get_settings;

/// Fluid Speech Pro on server-pc answered 2-20 s prefixes in 0.1-0.4 s, so a
/// request every 0.7 s keeps the preview within about a second of speech while
/// leaving the server idle most of the time for the final request.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(700);
/// 300 ms at 16 kHz: less new audio than this rarely changes the text.
const MIN_NEW_SAMPLES: usize = 4_800;
/// How long the worker waits for a frame before checking for a due preview
/// or a returned one. VAD drops silence, so frames can stop for a while.
const TICK: Duration = Duration::from_millis(50);

/// When the next preview is due, and which answers may still be shown.
/// Previews are identified by how many samples they covered: the buffer only
/// grows, so more samples means a newer preview.
struct PreviewSchedule {
    last_sent: Instant,
    sent_samples: usize,
    shown_samples: usize,
}

impl PreviewSchedule {
    fn new(now: Instant) -> Self {
        Self {
            last_sent: now,
            sent_samples: 0,
            shown_samples: 0,
        }
    }

    /// One request at a time, at most every [`PREVIEW_INTERVAL`], and only
    /// once [`MIN_NEW_SAMPLES`] arrived since the last one.
    fn due(&self, now: Instant, buffered: usize, in_flight: bool) -> bool {
        !in_flight
            && now.saturating_duration_since(self.last_sent) >= PREVIEW_INTERVAL
            && buffered >= self.sent_samples + MIN_NEW_SAMPLES
    }

    fn sent(&mut self, now: Instant, samples: usize) {
        self.last_sent = now;
        self.sent_samples = samples;
    }

    /// Whether an answer over `samples` may replace the text on screen: only
    /// one over more audio than what is shown.
    fn accept(&mut self, samples: usize) -> bool {
        if samples <= self.shown_samples {
            return false;
        }
        self.shown_samples = samples;
        true
    }
}

type PreviewResult = (usize, anyhow::Result<String>);

impl TranscriptionManager {
    /// Collects the recording's frames and keeps one preview request in
    /// flight over them, emitting each newer answer to the overlay. Finalize
    /// and cancel abort the request in flight instead of waiting for it.
    pub(super) fn run_cloud_preview_worker(
        &self,
        rx: mpsc::Receiver<StreamCmd>,
        cancelled: Arc<AtomicBool>,
        worker_id: u64,
    ) {
        let _worker = StreamWorkerGuard {
            worker_id,
            active_stream_worker: Arc::clone(&self.active_stream_worker),
            stream_active: Arc::clone(&self.stream_active),
        };

        let req = match cloud_request(&get_settings(&self.app_handle)) {
            Ok(req) => req,
            Err(e) => {
                warn!("Cloud preview unavailable: {}", e);
                self.router.clear();
                drain_until_finalize(rx);
                return;
            }
        };
        info!("Cloud preview started (model '{}')", req.model);
        self.stream_active.store(true, Ordering::Release);
        self.touch_activity();

        let (done_tx, done_rx) = mpsc::channel::<PreviewResult>();
        let mut audio: Vec<f32> = Vec::new();
        let mut schedule = PreviewSchedule::new(Instant::now());
        let mut in_flight: Option<JoinHandle<()>> = None;

        loop {
            match rx.recv_timeout(TICK) {
                Ok(StreamCmd::Feed(pcm)) => {
                    if cancelled.load(Ordering::Acquire) {
                        continue;
                    }
                    self.touch_activity();
                    audio.extend_from_slice(&pcm);
                }
                Ok(StreamCmd::Finalize { reply, .. }) => {
                    abort_preview(in_flight.take());
                    let _ = reply.send(Ok(None));
                    return;
                }
                Ok(StreamCmd::Cancel) | Err(RecvTimeoutError::Disconnected) => {
                    abort_preview(in_flight.take());
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {}
            }

            while let Ok((samples, result)) = done_rx.try_recv() {
                in_flight = None;
                match result {
                    Ok(text) if schedule.accept(samples) => {
                        emit_stream_text(&self.app_handle, "", &text)
                    }
                    Ok(_) => debug!("Dropped a stale cloud preview ({} samples)", samples),
                    Err(e) => debug!("Cloud preview failed: {}", e),
                }
            }

            let now = Instant::now();
            if cancelled.load(Ordering::Acquire)
                || !schedule.due(now, audio.len(), in_flight.is_some())
            {
                continue;
            }
            schedule.sent(now, audio.len());
            let req = req.clone();
            let snapshot = audio.clone();
            let done = done_tx.clone();
            in_flight = Some(tauri::async_runtime::spawn(async move {
                let result = crate::stt_cloud::transcribe(&req, &snapshot).await;
                let _ = done.send((snapshot.len(), result));
            }));
        }
    }
}

/// Drop the preview request in flight, closing its connection, so it neither
/// delays the final request nor reaches the overlay.
fn abort_preview(in_flight: Option<JoinHandle<()>>) {
    if let Some(handle) = in_flight {
        handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_preview_waits_for_the_interval_and_enough_audio() {
        let start = Instant::now();
        let schedule = PreviewSchedule::new(start);
        let later = start + PREVIEW_INTERVAL;
        assert!(!schedule.due(start, MIN_NEW_SAMPLES, false));
        assert!(!schedule.due(later, MIN_NEW_SAMPLES - 1, false));
        assert!(schedule.due(later, MIN_NEW_SAMPLES, false));
    }

    #[test]
    fn next_preview_needs_new_audio_and_no_request_in_flight() {
        let start = Instant::now();
        let mut schedule = PreviewSchedule::new(start);
        let first = start + PREVIEW_INTERVAL;
        schedule.sent(first, 10_000);
        let next = first + PREVIEW_INTERVAL;
        assert!(!schedule.due(next - Duration::from_millis(1), 20_000, false));
        assert!(!schedule.due(next, 10_000 + MIN_NEW_SAMPLES - 1, false));
        assert!(!schedule.due(next, 20_000, true));
        assert!(schedule.due(next, 10_000 + MIN_NEW_SAMPLES, false));
    }

    #[test]
    fn an_answer_over_older_audio_never_replaces_a_newer_one() {
        let mut schedule = PreviewSchedule::new(Instant::now());
        assert!(schedule.accept(20_000));
        assert!(!schedule.accept(10_000));
        assert!(!schedule.accept(20_000));
        assert!(schedule.accept(30_000));
    }
}
