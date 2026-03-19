//! Webhook alert dispatcher — POSTs JSON to configured URLs on alert conditions.
//!
//! Uses a bounded mpsc channel to decouple the measurement loop from HTTP latency.
//! A background thread drains the channel and POSTs to each configured URL.

use serde::Serialize;

/// JSON payload sent to webhook URLs.
#[derive(Debug, Clone, Serialize)]
pub struct WebhookPayload {
    pub source: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub device: String,
    pub alert: WebhookAlert,
}

/// Alert detail within a webhook payload.
#[derive(Debug, Clone, Serialize)]
pub struct WebhookAlert {
    pub level: String,
    pub rule: String,
    pub message: String,
}

/// Start the webhook dispatcher background thread.
///
/// Returns a sender for enqueueing payloads and the thread handle.
pub fn start_dispatcher(
    urls: Vec<String>,
) -> (
    std::sync::mpsc::SyncSender<WebhookPayload>,
    std::thread::JoinHandle<()>,
) {
    let (tx, rx) = std::sync::mpsc::sync_channel::<WebhookPayload>(64);

    let handle = std::thread::spawn(move || {
        for payload in rx {
            let json = match serde_json::to_string(&payload) {
                Ok(j) => j,
                Err(e) => {
                    tracing::warn!("webhook: failed to serialize payload: {e}");
                    continue;
                }
            };

            for url in &urls {
                match ureq::post(url)
                    .set("Content-Type", "application/json")
                    .send_string(&json)
                {
                    Ok(_) => {
                        tracing::debug!("webhook: posted alert to {url}");
                    }
                    Err(e) => {
                        tracing::warn!("webhook: POST to {url} failed: {e}");
                    }
                }
            }
        }
    });

    (tx, handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_payload_serialization() {
        let payload = WebhookPayload {
            source: "gpu-roofline".to_string(),
            timestamp: chrono::Utc::now(),
            device: "NVIDIA H100 SXM".to_string(),
            alert: WebhookAlert {
                level: "critical".to_string(),
                rule: "BandwidthDrop".to_string(),
                message: "Bandwidth dropped to 50% of baseline".to_string(),
            },
        };

        let json = serde_json::to_string_pretty(&payload).unwrap();
        assert!(json.contains("gpu-roofline"));
        assert!(json.contains("critical"));
        assert!(json.contains("BandwidthDrop"));
        assert!(json.contains("NVIDIA H100 SXM"));
    }

    #[test]
    fn test_dispatcher_handles_no_urls() {
        let (tx, _handle) = start_dispatcher(Vec::new());
        // Should accept payload without error even with no URLs
        let payload = WebhookPayload {
            source: "test".to_string(),
            timestamp: chrono::Utc::now(),
            device: "test".to_string(),
            alert: WebhookAlert {
                level: "warning".to_string(),
                rule: "Test".to_string(),
                message: "test".to_string(),
            },
        };
        tx.try_send(payload).unwrap();
        // Drop sender to let thread finish
        drop(tx);
    }
}
