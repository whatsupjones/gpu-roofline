//! TUI dashboard for vGPU lifecycle monitoring.
//!
//! Renders lifecycle events, instance status, contention indicators,
//! and teardown verification results in a terminal dashboard.

use std::time::Instant;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::prelude::*;
use ratatui::widgets::*;

use gpu_harness::vgpu::{VgpuEvent, VgpuEventType};

use super::vgpu_alerting::VgpuAlert;

/// Maximum event log entries to keep.
const MAX_EVENT_LOG: usize = 100;

/// State for the vGPU TUI dashboard.
pub struct VgpuTuiState {
    /// Event log (recent lifecycle events).
    pub event_log: Vec<EventLogEntry>,
    /// Active instance count over time.
    pub instance_history: Vec<u64>,
    /// Alert log.
    pub alert_log: Vec<String>,
    /// Current active instance count.
    pub active_count: u32,
    /// Total events received.
    pub total_events: u32,
    /// Total alerts triggered.
    pub total_alerts: u32,
    /// Session start time.
    pub start_time: Instant,
    /// Scenario name (if simulated).
    pub scenario_name: Option<String>,
    /// Whether to exit.
    pub should_quit: bool,
}

/// A single entry in the event log.
pub struct EventLogEntry {
    pub timestamp: String,
    pub event_type: String,
    pub instance_id: String,
    pub detail: String,
    pub style: Style,
}

impl VgpuTuiState {
    pub fn new(scenario_name: Option<String>) -> Self {
        Self {
            event_log: Vec::new(),
            instance_history: Vec::new(),
            alert_log: Vec::new(),
            active_count: 0,
            total_events: 0,
            total_alerts: 0,
            start_time: Instant::now(),
            scenario_name,
            should_quit: false,
        }
    }

    /// Push a lifecycle event into the TUI state.
    pub fn push_event(&mut self, event: &VgpuEvent, alerts: &[VgpuAlert]) {
        self.total_events += 1;

        let (event_type, detail, style) = match &event.event_type {
            VgpuEventType::Created {
                instance,
                spin_up_latency_ms,
            } => {
                self.active_count += 1;
                let latency = spin_up_latency_ms
                    .map(|ms| format!(" ({:.0}ms)", ms))
                    .unwrap_or_default();
                (
                    "CREATED".to_string(),
                    format!("{}{}", instance.name, latency),
                    Style::default().fg(Color::Green),
                )
            }
            VgpuEventType::Active { .. } => (
                "ACTIVE".to_string(),
                "transitioned to active".to_string(),
                Style::default().fg(Color::Cyan),
            ),
            VgpuEventType::Sampled { .. } => (
                "SAMPLE".to_string(),
                "periodic sample".to_string(),
                Style::default().fg(Color::DarkGray),
            ),
            VgpuEventType::ContentionDetected {
                affected_instances,
                bandwidth_impact,
                caused_by,
                ..
            } => {
                let drops: Vec<String> = affected_instances
                    .iter()
                    .zip(bandwidth_impact.iter())
                    .map(|(id, impact)| format!("{}: -{:.0}%", id, (1.0 - impact) * 100.0))
                    .collect();
                (
                    "CONTENTION".to_string(),
                    format!("caused by {}: {}", caused_by, drops.join(", ")),
                    Style::default().fg(Color::Yellow).bold(),
                )
            }
            VgpuEventType::TeardownStarted { .. } => (
                "TEARDOWN".to_string(),
                "teardown initiated".to_string(),
                Style::default().fg(Color::Yellow),
            ),
            VgpuEventType::Destroyed { verification, .. } => {
                self.active_count = self.active_count.saturating_sub(1);
                let status = if verification.memory_reclaimed {
                    "clean".to_string()
                } else {
                    let ghost_mb = verification.ghost_allocations_bytes as f64 / (1024.0 * 1024.0);
                    format!("GHOST {:.0}MB", ghost_mb)
                };
                let style = if verification.memory_reclaimed {
                    Style::default().fg(Color::Blue)
                } else {
                    Style::default().fg(Color::Red).bold()
                };
                ("DESTROYED".to_string(), status, style)
            }
        };

        let entry = EventLogEntry {
            timestamp: event.timestamp.format("%H:%M:%S%.3f").to_string(),
            event_type,
            instance_id: event.instance_id.clone(),
            detail,
            style,
        };

        self.event_log.push(entry);
        if self.event_log.len() > MAX_EVENT_LOG {
            self.event_log.remove(0);
        }

        // Track instance count history
        if self.instance_history.len() >= 60 {
            self.instance_history.remove(0);
        }
        self.instance_history.push(self.active_count as u64);

        // Log alerts
        for alert in alerts {
            self.total_alerts += 1;
            let level_str = match alert.level {
                super::alerting::AlertLevel::Warning => "WARN",
                super::alerting::AlertLevel::Critical => "CRIT",
            };
            self.alert_log.push(format!(
                "[{}] {} [{}] {}",
                event.timestamp.format("%H:%M:%S"),
                level_str,
                alert.rule,
                alert.message
            ));
        }
    }
}

/// Check for keyboard input (non-blocking). Returns true if user wants to quit.
pub fn poll_quit() -> bool {
    if event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Press {
                return matches!(key.code, KeyCode::Char('q') | KeyCode::Esc);
            }
        }
    }
    false
}

/// Render the vGPU lifecycle TUI dashboard.
pub fn draw(frame: &mut Frame, state: &VgpuTuiState) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Header
            Constraint::Min(0),    // Main content
            Constraint::Length(1), // Footer
        ])
        .split(area);

    draw_header(frame, chunks[0], state);
    draw_main(frame, chunks[1], state);
    draw_footer(frame, chunks[2], state);
}

fn draw_header(frame: &mut Frame, area: Rect, state: &VgpuTuiState) {
    let scenario = state.scenario_name.as_deref().unwrap_or("live hardware");

    let header = Line::from(vec![
        Span::styled(" gpu-roofline vgpu ", Style::default().bold()),
        Span::raw("── "),
        Span::styled(scenario, Style::default().fg(Color::Cyan)),
        Span::raw("  "),
        Span::styled(
            format!("{} active", state.active_count),
            Style::default().fg(if state.active_count > 0 {
                Color::Green
            } else {
                Color::DarkGray
            }),
        ),
        Span::raw("  "),
        Span::styled(
            format!("{} events", state.total_events),
            Style::default().fg(Color::White),
        ),
        Span::raw("  "),
        Span::styled(
            format!("{} alerts", state.total_alerts),
            Style::default().fg(if state.total_alerts > 0 {
                Color::Red
            } else {
                Color::DarkGray
            }),
        ),
    ]);

    frame.render_widget(Paragraph::new(header), area);
}

fn draw_footer(frame: &mut Frame, area: Rect, state: &VgpuTuiState) {
    let uptime = state.start_time.elapsed();
    let mins = uptime.as_secs() / 60;
    let secs = uptime.as_secs() % 60;

    let footer = Line::from(vec![
        Span::styled(" q", Style::default().fg(Color::Yellow)),
        Span::raw(" quit  "),
        Span::raw(format!("Uptime: {}m {:02}s", mins, secs)),
    ]);

    frame.render_widget(
        Paragraph::new(footer).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}

fn draw_main(frame: &mut Frame, area: Rect, state: &VgpuTuiState) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);

    // Left: Event log
    draw_event_log(frame, cols[0], state);

    // Right: Instance sparkline + alerts
    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(3)])
        .split(cols[1]);

    draw_instances(frame, right_rows[0], state);
    draw_alerts(frame, right_rows[1], state);
}

fn draw_event_log(frame: &mut Frame, area: Rect, state: &VgpuTuiState) {
    let block = Block::default()
        .title(" Lifecycle Events ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if state.event_log.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                " Waiting for events...",
                Style::default().fg(Color::DarkGray),
            )),
            inner,
        );
        return;
    }

    let visible = inner.height as usize;
    let start = state.event_log.len().saturating_sub(visible);
    let lines: Vec<Line> = state.event_log[start..]
        .iter()
        .map(|entry| {
            Line::from(vec![
                Span::styled(
                    format!(" {} ", entry.timestamp),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(format!("{:<11}", entry.event_type), entry.style),
                Span::styled(
                    format!("{:<12}", entry.instance_id),
                    Style::default().fg(Color::White),
                ),
                Span::styled(&entry.detail, Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();

    frame.render_widget(Paragraph::new(lines), inner);
}

fn draw_instances(frame: &mut Frame, area: Rect, state: &VgpuTuiState) {
    let block = Block::default()
        .title(" Active Instances ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Sparkline of instance count
    if !state.instance_history.is_empty() {
        let sparkline = Sparkline::default()
            .data(&state.instance_history)
            .style(Style::default().fg(Color::Green));
        frame.render_widget(sparkline, inner);
    } else {
        frame.render_widget(
            Paragraph::new(Span::styled(
                " No data",
                Style::default().fg(Color::DarkGray),
            )),
            inner,
        );
    }
}

fn draw_alerts(frame: &mut Frame, area: Rect, state: &VgpuTuiState) {
    let block = Block::default()
        .title(" Alerts ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if state.alert_log.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                " No alerts",
                Style::default().fg(Color::DarkGray),
            )),
            inner,
        );
    } else {
        let visible = inner.height as usize;
        let start = state.alert_log.len().saturating_sub(visible);
        let lines: Vec<Line> = state.alert_log[start..]
            .iter()
            .map(|msg| {
                let color = if msg.contains("CRIT") {
                    Color::Red
                } else {
                    Color::Yellow
                };
                Line::from(Span::styled(format!(" {msg}"), Style::default().fg(color)))
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }
}
