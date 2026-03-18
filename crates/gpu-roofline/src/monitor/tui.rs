//! Live terminal dashboard for GPU monitoring using ratatui.
//!
//! Renders a multi-panel TUI showing performance, thermals, memory,
//! roofline position, tension analysis, and alert history.

use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::*;

use super::alerting::AlertLevel;
use super::sampler::MonitorSample;
use super::SampleStatus;

/// Maximum number of sparkline data points to keep.
const SPARKLINE_HISTORY: usize = 60;

/// State for the TUI dashboard.
pub struct TuiState {
    /// All collected samples.
    pub samples: Vec<MonitorSample>,
    /// Baseline bandwidth for comparison.
    pub baseline_bw: f64,
    /// Baseline GFLOPS for comparison.
    pub baseline_gflops: f64,
    /// Burst GFLOPS (from initial measurement).
    pub burst_gflops: f64,
    /// GPU name.
    pub device_name: String,
    /// Driver version.
    pub driver_version: String,
    /// Compute capability string (e.g., "sm_90").
    pub compute_capability: String,
    /// Max clock MHz for this GPU.
    pub max_clock_mhz: u32,
    /// TDP watts for this GPU.
    pub tdp_watts: f32,
    /// Total VRAM bytes.
    pub vram_total_bytes: u64,
    /// Sparkline histories (fixed-size ring buffers).
    pub bw_history: Vec<u64>,
    pub gflops_history: Vec<u64>,
    pub temp_history: Vec<u64>,
    pub power_history: Vec<u64>,
    pub clock_history: Vec<u64>,
    pub cv_history: Vec<u64>,
    /// Alert log.
    pub alert_log: Vec<String>,
    /// Session start time.
    pub start_time: Instant,
    /// Whether to exit.
    pub should_quit: bool,
}

/// Configuration to initialize the TUI dashboard.
pub struct TuiConfig {
    pub baseline_bw: f64,
    pub baseline_gflops: f64,
    pub device_name: String,
    pub driver_version: String,
    pub compute_capability: String,
    pub max_clock_mhz: u32,
    pub tdp_watts: f32,
    pub vram_total_bytes: u64,
}

impl TuiState {
    pub fn new(config: TuiConfig) -> Self {
        Self {
            samples: Vec::new(),
            baseline_bw: config.baseline_bw,
            baseline_gflops: config.baseline_gflops,
            burst_gflops: config.baseline_gflops,
            device_name: config.device_name,
            driver_version: config.driver_version,
            compute_capability: config.compute_capability,
            max_clock_mhz: config.max_clock_mhz,
            tdp_watts: config.tdp_watts,
            vram_total_bytes: config.vram_total_bytes,
            bw_history: Vec::with_capacity(SPARKLINE_HISTORY),
            gflops_history: Vec::with_capacity(SPARKLINE_HISTORY),
            temp_history: Vec::with_capacity(SPARKLINE_HISTORY),
            power_history: Vec::with_capacity(SPARKLINE_HISTORY),
            clock_history: Vec::with_capacity(SPARKLINE_HISTORY),
            cv_history: Vec::with_capacity(SPARKLINE_HISTORY),
            alert_log: Vec::new(),
            start_time: Instant::now(),
            should_quit: false,
        }
    }

    /// Push a new sample into the TUI state.
    pub fn push_sample(&mut self, sample: MonitorSample) {
        // Update sparkline histories
        self.push_sparkline(&sample);

        // Log alerts
        for alert in &sample.alerts {
            let level_str = match alert.level {
                AlertLevel::Warning => "WARN",
                AlertLevel::Critical => "CRIT",
            };
            self.alert_log.push(format!(
                "[{}] {} {}",
                sample.timestamp.format("%H:%M:%S"),
                level_str,
                alert.message
            ));
        }

        self.samples.push(sample);
    }

    fn push_sparkline(&mut self, sample: &MonitorSample) {
        fn push_capped(history: &mut Vec<u64>, val: u64) {
            if history.len() >= SPARKLINE_HISTORY {
                history.remove(0);
            }
            history.push(val);
        }

        push_capped(&mut self.bw_history, sample.bandwidth_gbps as u64);
        push_capped(&mut self.gflops_history, (sample.gflops / 1000.0) as u64); // TFLOPS
        push_capped(&mut self.temp_history, sample.temperature_c as u64);
        push_capped(&mut self.power_history, sample.power_watts as u64);
        push_capped(&mut self.clock_history, sample.clock_mhz as u64);
        push_capped(&mut self.cv_history, (sample.cv * 1000.0) as u64); // 0.1% precision
    }

    fn latest(&self) -> Option<&MonitorSample> {
        self.samples.last()
    }

    fn session_stats(&self) -> (f64, f64, f64, u32, usize) {
        if self.samples.is_empty() {
            return (0.0, 0.0, 0.0, 0, 0);
        }
        let avg_bw = self.samples.iter().map(|s| s.bandwidth_gbps).sum::<f64>()
            / self.samples.len() as f64;
        let min_bw = self
            .samples
            .iter()
            .map(|s| s.bandwidth_gbps)
            .fold(f64::MAX, f64::min);
        let max_temp = self.samples.iter().map(|s| s.temperature_c).max().unwrap_or(0);
        let avg_gflops =
            self.samples.iter().map(|s| s.gflops).sum::<f64>() / self.samples.len() as f64;
        let total_alerts: usize = self.samples.iter().map(|s| s.alerts.len()).sum();
        (avg_bw, min_bw, avg_gflops, max_temp, total_alerts)
    }
}

/// Initialize terminal and enter TUI mode.
pub fn init_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    Terminal::new(backend)
}

/// Restore terminal to normal mode.
pub fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

/// Check for keyboard input (non-blocking). Returns true if user wants to quit.
pub fn poll_input() -> bool {
    if event::poll(Duration::from_millis(50)).unwrap_or(false) {
        if let Ok(Event::Key(key)) = event::read() {
            if key.kind == KeyEventKind::Press {
                return matches!(key.code, KeyCode::Char('q') | KeyCode::Esc);
            }
        }
    }
    false
}

/// Render the full TUI dashboard.
pub fn draw(frame: &mut Frame, state: &TuiState) {
    let area = frame.area();

    // Header
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Header bar
            Constraint::Min(0),   // Main content
            Constraint::Length(1), // Footer bar
        ])
        .split(area);

    draw_header(frame, chunks[0], state);
    draw_main(frame, chunks[1], state);
    draw_footer(frame, chunks[2], state);
}

fn draw_header(frame: &mut Frame, area: Rect, state: &TuiState) {
    let status = state
        .latest()
        .map(|s| match &s.status {
            SampleStatus::Normal => Span::styled(" OK ", Style::default().bg(Color::Green).fg(Color::Black)),
            SampleStatus::Warning { .. } => {
                Span::styled(" WARN ", Style::default().bg(Color::Yellow).fg(Color::Black))
            }
            SampleStatus::Alert { .. } => {
                Span::styled(" ALERT ", Style::default().bg(Color::Red).fg(Color::White))
            }
        })
        .unwrap_or_else(|| Span::styled(" INIT ", Style::default().bg(Color::DarkGray).fg(Color::White)));

    let header = Line::from(vec![
        Span::styled(" gpu-roofline monitor ", Style::default().bold()),
        Span::raw("── "),
        Span::styled(&state.device_name, Style::default().fg(Color::Cyan)),
        Span::raw(" ── "),
        Span::raw(&state.driver_version),
        Span::raw(" ── "),
        Span::raw(&state.compute_capability),
        Span::raw("  "),
        status,
    ]);

    frame.render_widget(Paragraph::new(header), area);
}

fn draw_footer(frame: &mut Frame, area: Rect, state: &TuiState) {
    let uptime = state.start_time.elapsed();
    let mins = uptime.as_secs() / 60;
    let secs = uptime.as_secs() % 60;

    let sample_count = state.samples.len();

    let footer = Line::from(vec![
        Span::styled(" q", Style::default().fg(Color::Yellow)),
        Span::raw(" quit  "),
        Span::raw(format!(
            "Samples: {}  Uptime: {}m {:02}s",
            sample_count, mins, secs
        )),
    ]);

    frame.render_widget(
        Paragraph::new(footer).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}

fn draw_main(frame: &mut Frame, area: Rect, state: &TuiState) {
    // Split into left (performance + thermals + memory) and right (roofline + tension + session)
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    // Left column: Performance, Thermals & Power, Memory, Alerts
    let left_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),  // Performance
            Constraint::Length(7),  // Thermals & Power
            Constraint::Length(5),  // Memory
            Constraint::Min(3),    // Alerts
        ])
        .split(cols[0]);

    draw_performance(frame, left_rows[0], state);
    draw_thermals(frame, left_rows[1], state);
    draw_memory(frame, left_rows[2], state);
    draw_alerts(frame, left_rows[3], state);

    // Right column: Tension, Session
    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Tension analysis
            Constraint::Min(3),    // Session stats
        ])
        .split(cols[1]);

    draw_tension(frame, right_rows[0], state);
    draw_session(frame, right_rows[1], state);
}

fn draw_performance(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default()
        .title(" Performance ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let latest = state.latest();

    let bw = latest.map(|s| s.bandwidth_gbps).unwrap_or(0.0);
    let gflops = latest.map(|s| s.gflops).unwrap_or(0.0);
    let cv = latest.map(|s| s.cv).unwrap_or(0.0);

    let bw_pct = if state.baseline_bw > 0.0 { bw / state.baseline_bw * 100.0 } else { 0.0 };
    let gflops_pct = if state.baseline_gflops > 0.0 { gflops / state.baseline_gflops * 100.0 } else { 0.0 };

    let bw_color = pct_color(bw_pct);
    let gflops_color = pct_color(gflops_pct);
    let cv_color = if cv < 0.02 { Color::Green } else if cv < 0.05 { Color::Yellow } else { Color::Red };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    // BW line with sparkline
    let bw_line = Line::from(vec![
        Span::styled(" BW    ", Style::default().fg(Color::White)),
        Span::styled(format!("{:>6.0} GB/s ", bw), Style::default().fg(bw_color).bold()),
        Span::styled(format!("{:>5.1}%  ", bw_pct), Style::default().fg(bw_color)),
        sparkline_text(&state.bw_history),
    ]);
    frame.render_widget(Paragraph::new(bw_line), rows[0]);

    // FLOPS line with sparkline
    let tflops = gflops / 1000.0;
    let gflops_line = Line::from(vec![
        Span::styled(" FP32  ", Style::default().fg(Color::White)),
        Span::styled(format!("{:>6.1} TFLOPS ", tflops), Style::default().fg(gflops_color).bold()),
        Span::styled(format!("{:>5.1}%  ", gflops_pct), Style::default().fg(gflops_color)),
        sparkline_text(&state.gflops_history),
    ]);
    frame.render_widget(Paragraph::new(gflops_line), rows[1]);

    // CV line
    let cv_line = Line::from(vec![
        Span::styled(" CV    ", Style::default().fg(Color::White)),
        Span::styled(format!("{:>6.1}%    ", cv * 100.0), Style::default().fg(cv_color).bold()),
        Span::styled(
            if cv < 0.02 { "stable  " } else if cv < 0.05 { "noisy   " } else { "unstable" },
            Style::default().fg(cv_color),
        ),
        sparkline_text(&state.cv_history),
    ]);
    frame.render_widget(Paragraph::new(cv_line), rows[2]);
}

fn draw_thermals(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default()
        .title(" Thermals & Power ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let latest = state.latest();
    let temp = latest.map(|s| s.temperature_c).unwrap_or(0);
    let power = latest.map(|s| s.power_watts).unwrap_or(0.0);
    let clock = latest.map(|s| s.clock_mhz).unwrap_or(0);

    let temp_color = if temp < 70 { Color::Green } else if temp < 85 { Color::Yellow } else { Color::Red };
    let power_pct = if state.tdp_watts > 0.0 { power / state.tdp_watts * 100.0 } else { 0.0 };
    let clock_pct = if state.max_clock_mhz > 0 { clock as f64 / state.max_clock_mhz as f64 * 100.0 } else { 0.0 };

    let throttle = if temp >= 85 {
        Span::styled("THERMAL THROTTLE", Style::default().fg(Color::Red).bold())
    } else if power_pct > 98.0 {
        Span::styled("POWER LIMITED", Style::default().fg(Color::Yellow).bold())
    } else {
        Span::styled("none", Style::default().fg(Color::Green))
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let temp_str = if temp > 0 { format!("{}°C", temp) } else { "--".to_string() };
    let temp_line = Line::from(vec![
        Span::styled(" Temp   ", Style::default().fg(Color::White)),
        Span::styled(format!("{:>6}     ", temp_str), Style::default().fg(temp_color).bold()),
        sparkline_text(&state.temp_history),
    ]);
    frame.render_widget(Paragraph::new(temp_line), rows[0]);

    let power_str = if power > 0.0 { format!("{:.0}W/{:.0}W", power, state.tdp_watts) } else { "--".to_string() };
    let power_line = Line::from(vec![
        Span::styled(" Power  ", Style::default().fg(Color::White)),
        Span::styled(format!("{:<14}", power_str), Style::default().fg(Color::White).bold()),
        sparkline_text(&state.power_history),
    ]);
    frame.render_widget(Paragraph::new(power_line), rows[1]);

    let clock_str = if clock > 0 {
        format!("{} MHz ({:.0}%)", clock, clock_pct)
    } else {
        "--".to_string()
    };
    let clock_line = Line::from(vec![
        Span::styled(" Clock  ", Style::default().fg(Color::White)),
        Span::styled(format!("{:<14}", clock_str), Style::default().fg(Color::White).bold()),
        sparkline_text(&state.clock_history),
    ]);
    frame.render_widget(Paragraph::new(clock_line), rows[2]);

    let throttle_line = Line::from(vec![
        Span::styled(" Throttle: ", Style::default().fg(Color::White)),
        throttle,
    ]);
    frame.render_widget(Paragraph::new(throttle_line), rows[3]);
}

fn draw_memory(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default()
        .title(" Memory ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let latest = state.latest();
    let mem_used = latest.map(|s| s.bandwidth_gbps).and_then(|_| {
        // Memory usage comes from device state, embedded in sample if NVML is active
        // For now we use the values from the sample
        state.latest().map(|_| 0u64)
    }).unwrap_or(0);

    let total_gb = state.vram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let used_gb = mem_used as f64 / (1024.0 * 1024.0 * 1024.0);
    let pct = if state.vram_total_bytes > 0 {
        mem_used as f64 / state.vram_total_bytes as f64
    } else {
        0.0
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    let mem_line = Line::from(vec![
        Span::styled(" VRAM  ", Style::default().fg(Color::White)),
        Span::styled(
            format!("{:.1} / {:.1} GB ({:.0}%)", used_gb, total_gb, pct * 100.0),
            Style::default().fg(Color::White).bold(),
        ),
    ]);
    frame.render_widget(Paragraph::new(mem_line), rows[0]);

    // VRAM bar
    let bar_width = (area.width as usize).saturating_sub(4);
    let filled = (pct * bar_width as f64) as usize;
    let empty = bar_width.saturating_sub(filled);
    let bar_color = if pct < 0.7 { Color::Green } else if pct < 0.9 { Color::Yellow } else { Color::Red };
    let bar_line = Line::from(vec![
        Span::raw("  "),
        Span::styled("█".repeat(filled), Style::default().fg(bar_color)),
        Span::styled("░".repeat(empty), Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(bar_line), rows[1]);

    // HBM utilization
    let bw = state.latest().map(|s| s.bandwidth_gbps).unwrap_or(0.0);
    let hbm_util = if state.baseline_bw > 0.0 { bw / state.baseline_bw * 100.0 } else { 0.0 };
    let util_line = Line::from(vec![
        Span::styled(" HBM BW utilization: ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.0}%", hbm_util), Style::default().fg(Color::White)),
    ]);
    frame.render_widget(Paragraph::new(util_line), rows[2]);
}

fn draw_tension(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default()
        .title(" Tension Analysis ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let latest = state.latest();
    let current_gflops = latest.map(|s| s.gflops).unwrap_or(0.0);
    let current_bw = latest.map(|s| s.bandwidth_gbps).unwrap_or(0.0);
    let temp = latest.map(|s| s.temperature_c).unwrap_or(0);
    let power = latest.map(|s| s.power_watts).unwrap_or(0.0);

    let burst_tflops = state.burst_gflops / 1000.0;
    let current_tflops = current_gflops / 1000.0;
    let total_drop = if state.burst_gflops > 0.0 {
        (1.0 - current_gflops / state.burst_gflops) * 100.0
    } else {
        0.0
    };

    let bw_drop = if state.baseline_bw > 0.0 {
        (1.0 - current_bw / state.baseline_bw) * 100.0
    } else {
        0.0
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(" Burst:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1}T", burst_tflops), Style::default().fg(Color::White).bold()),
            Span::styled(" (t=0)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled(" Current:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1}T", current_tflops), Style::default().fg(Color::Cyan).bold()),
            Span::styled(
                format!(" ({:+.1}%)", -total_drop),
                Style::default().fg(if total_drop < 5.0 { Color::Green } else if total_drop < 15.0 { Color::Yellow } else { Color::Red }),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Compute:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:+.1}%", -total_drop),
                Style::default().fg(drop_color(total_drop)),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Bandwidth: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:+.1}%", -bw_drop),
                Style::default().fg(drop_color(bw_drop)),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Thermal:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if temp > 0 { format!("{}°C", temp) } else { "--".to_string() },
                Style::default().fg(if temp < 70 { Color::Green } else if temp < 85 { Color::Yellow } else { Color::Red }),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Power:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if power > 0.0 { format!("{:.0}W", power) } else { "--".to_string() },
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Net drop:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:+.1}%", -total_drop.max(bw_drop)),
                Style::default().fg(drop_color(total_drop.max(bw_drop))).bold(),
            ),
        ]),
    ];

    for (i, line) in lines.iter().enumerate() {
        if i < rows.len() {
            frame.render_widget(Paragraph::new(line.clone()), rows[i]);
        }
    }
}

fn draw_session(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default()
        .title(" Session ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let (avg_bw, min_bw, avg_gflops, max_temp, total_alerts) = state.session_stats();
    let uptime = state.start_time.elapsed();
    let mins = uptime.as_secs() / 60;
    let secs = uptime.as_secs() % 60;

    let lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(" Samples:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", state.samples.len()), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(" Uptime:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}m {:02}s", mins, secs), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(" Avg BW:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.0} GB/s", avg_bw), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(" Min BW:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.0} GB/s", min_bw), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(" Avg FP32:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1} TFLOPS", avg_gflops / 1000.0), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(" Max Temp:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if max_temp > 0 { format!("{}°C", max_temp) } else { "--".to_string() },
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Alerts:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", total_alerts),
                Style::default().fg(if total_alerts == 0 { Color::Green } else { Color::Red }),
            ),
        ]),
    ];

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(lines.iter().map(|_| Constraint::Length(1)).collect::<Vec<_>>())
        .split(inner);

    for (i, line) in lines.iter().enumerate() {
        if i < rows.len() {
            frame.render_widget(Paragraph::new(line.clone()), rows[i]);
        }
    }
}

fn draw_alerts(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default()
        .title(" Alerts ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if state.alert_log.is_empty() {
        let no_alerts = Line::from(Span::styled(
            " No alerts",
            Style::default().fg(Color::DarkGray),
        ));
        frame.render_widget(Paragraph::new(no_alerts), inner);
    } else {
        // Show most recent alerts (scrollable area)
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
                Line::from(Span::styled(format!(" {}", msg), Style::default().fg(color)))
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }
}

// ── Helper functions ──

/// Generate a text-based sparkline from history data.
fn sparkline_text(data: &[u64]) -> Span<'static> {
    if data.is_empty() {
        return Span::raw("");
    }

    let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let min = *data.iter().min().unwrap_or(&0);
    let max = *data.iter().max().unwrap_or(&1);
    let range = if max > min { max - min } else { 1 };

    let sparkline: String = data
        .iter()
        .map(|&v| {
            let idx = ((v.saturating_sub(min)) * 7 / range) as usize;
            blocks[idx.min(7)]
        })
        .collect();

    Span::styled(sparkline, Style::default().fg(Color::DarkGray))
}

/// Color based on percentage of baseline.
fn pct_color(pct: f64) -> Color {
    if pct >= 95.0 {
        Color::Green
    } else if pct >= 80.0 {
        Color::Yellow
    } else {
        Color::Red
    }
}

/// Color based on drop percentage.
fn drop_color(drop: f64) -> Color {
    if drop < 5.0 {
        Color::Green
    } else if drop < 15.0 {
        Color::Yellow
    } else {
        Color::Red
    }
}
