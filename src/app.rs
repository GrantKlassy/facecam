use eframe::egui::{self, Color32, Pos2, Rect};
use eframe::epaint::Vertex;
use ringbuf::traits::Consumer;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::audio::{AudioConsumer, AudioControl};
use crate::colors::{Palette, BUMBLEBEE, PALETTES};
use crate::fft::Analyzer;
use crate::nowplaying::{SharedTrack, Track};

pub const NUM_BARS: usize = 64;
pub const FFT_SIZE: usize = 4096;
pub const SAMPLE_RATE: u32 = 44100;
pub const LOW_HZ: f32 = 30.0;
pub const HIGH_HZ: f32 = 16_000.0;

// Heartbeat spring. Underdamped so a beat snaps the heart up, overshoots, and
// settles — the elastic "thump" of a pulse. `LUB`/`DUB` are the two impulses of
// a single heartbeat; the dub is softer and lands ~0.14 s after the lub.
const HEART_K: f32 = 185.0;
const HEART_C: f32 = 9.0;
const LUB_IMPULSE: f32 = 3.4;
const DUB_IMPULSE: f32 = 2.0;

#[derive(Clone, Copy)]
enum Mode {
    Full,
    Half,
    Mirror,
    Rainbow,
}

const MODES: &[Mode] = &[Mode::Full, Mode::Half, Mode::Mirror, Mode::Rainbow];

impl Mode {
    fn name(&self) -> &'static str {
        match self {
            Mode::Full => "full",
            Mode::Half => "half",
            Mode::Mirror => "mirror",
            Mode::Rainbow => "rainbow",
        }
    }
}

#[derive(Clone, Copy)]
enum BarSide {
    Bottom,
    Top,
}

/// A sonar ring emitted from the heart on a beat; expands outward and fades.
#[derive(Clone, Copy)]
struct PulseRing {
    age: f32,
    max_age: f32,
    r0: f32,
    speed: f32,
}

pub struct FacecamApp {
    consumer: AudioConsumer,
    audio_control: AudioControl,
    analyzer: Analyzer,
    nowplaying: SharedTrack,
    last_track: Option<Track>,
    palette_idx: usize,
    mode_idx: usize,
    scratch: Vec<f32>,
    show_overlay: bool,
    show_controls: bool,
    screenshot_path: Option<std::path::PathBuf>,
    screenshot_counter: AtomicUsize,
    start_time: std::time::Instant,
    phase_offset: f32,

    // ── reactive visual state ──
    /// `FACECAM_DEMO=1`: synthesize a musical signal instead of reading capture,
    /// so the visualizer is lively with no audio routed (handy for the README).
    demo: bool,
    demo_clock: f64,
    rng: u32,
    /// Overall smoothed loudness, sets the heart's resting size.
    level: f32,
    /// Running average of bass energy, for beat detection.
    bass_avg: f32,
    /// Decaying 0..1 envelope kicked to 1.0 on each detected beat (drives glow).
    beat_env: f32,
    /// Seconds (since start) of the last detected beat, for refractory gating.
    last_beat: f32,
    /// Heart scale spring: `heart_x` is the offset from rest, `heart_v` its
    /// velocity. A beat injects an impulse; the underdamped spring gives the
    /// snap-and-settle of a real "lub". `dub_at` schedules the softer "dub".
    heart_x: f32,
    heart_v: f32,
    dub_at: f32,
    /// Time of the next *resting* heartbeat — fires only when the music isn't
    /// supplying beats, so the facecam always has a pulse.
    idle_at: f32,
    rings: Vec<PulseRing>,
}

impl FacecamApp {
    pub fn new(
        consumer: AudioConsumer,
        audio_control: AudioControl,
        nowplaying: SharedTrack,
    ) -> Self {
        let screenshot_path = std::env::var_os("FACECAM_SCREENSHOT").map(std::path::PathBuf::from);
        let mode_idx = std::env::var("FACECAM_MODE")
            .ok()
            .and_then(|name| MODES.iter().position(|m| m.name() == name.to_lowercase()))
            .unwrap_or(0);
        Self {
            consumer,
            audio_control,
            analyzer: Analyzer::new(FFT_SIZE, SAMPLE_RATE, NUM_BARS, LOW_HZ, HIGH_HZ),
            nowplaying,
            last_track: None,
            palette_idx: 0,
            mode_idx,
            scratch: vec![0.0; 8192],
            show_overlay: true,
            show_controls: false,
            screenshot_path,
            screenshot_counter: AtomicUsize::new(0),
            start_time: std::time::Instant::now(),
            phase_offset: std::env::var("FACECAM_PHASE_OFFSET")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            demo: std::env::var("FACECAM_DEMO").is_ok_and(|v| v != "0" && !v.is_empty()),
            demo_clock: 0.0,
            rng: 0x9e3779b9,
            level: 0.0,
            bass_avg: 0.0,
            beat_env: 0.0,
            last_beat: -1.0,
            heart_x: 0.0,
            heart_v: 0.0,
            dub_at: f32::MAX,
            idle_at: 1.0,
            rings: Vec::new(),
        }
    }

    /// Fast xorshift PRNG → f32 in [0, 1). Avoids pulling in the `rand` crate
    /// for the handful of random draws the particle system needs.
    fn rand(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Synthesize a musical mono signal (kick + hats + sweeping lead + air)
    /// and feed it to the analyzer, advancing an internal sample clock by `dt`.
    fn feed_demo_signal(&mut self, dt: f32) {
        use std::f64::consts::TAU;
        let sr = SAMPLE_RATE as f64;
        let want = ((dt as f64) * sr).clamp(0.0, 4096.0) as usize;
        let mut buf = [0.0f32; 4096];
        for slot in buf.iter_mut().take(want) {
            let t = self.demo_clock / sr;
            let beat_period = 0.46;
            let bt = (t % beat_period) / beat_period;
            let kick = (-(bt * 9.0)).exp();
            let bass = ((TAU * 52.0 * t).sin() + 0.5 * (TAU * 104.0 * t).sin()) * kick;
            let hat_env = (-(((t * 8.0) % 1.0) * 40.0)).exp();
            let hat = ((self.rand() * 2.0 - 1.0) as f64) * hat_env * 0.25;
            let lead_f = 220.0 * (1.0 + 0.5 * (TAU * 0.13 * t).sin());
            let lead = (TAU * lead_f * t).sin() * (0.12 + 0.08 * (TAU * 0.5 * t).sin());
            let mid = (TAU * 330.0 * t).sin() * 0.08 * (0.5 + 0.5 * (TAU * 0.25 * t).sin());
            let air = (self.rand() * 2.0 - 1.0) as f64 * 0.03;
            *slot = ((bass * 0.6 + lead * 0.5 + mid + hat + air) * 0.6) as f32;
            self.demo_clock += 1.0;
        }
        self.analyzer.ingest(&buf[..want]);
    }

    /// Track overall level, detect beats on the bass, drive the heartbeat spring
    /// (a "lub" impulse now, a softer "dub" scheduled just after), emit sonar
    /// rings, and age the existing ones.
    fn update_reactive(&mut self, dt: f32, rect: Rect) {
        let (bass, level) = {
            let bars = &self.analyzer.bars;
            let n = bars.len().max(1);
            let bass = bars[0..(n / 5).max(1)].iter().sum::<f32>() / (n / 5).max(1) as f32;
            let level = bars.iter().sum::<f32>() / n as f32;
            (bass, level)
        };
        // Fast attack, slow release for a level that swells then eases.
        let k = if level > self.level { 0.5 } else { 0.1 };
        self.level += (level - self.level) * k;

        self.bass_avg += (bass - self.bass_avg) * 0.08;
        let now = self.start_time.elapsed().as_secs_f32();
        if bass > (self.bass_avg * 1.45).max(0.22) && (now - self.last_beat) > 0.12 {
            self.last_beat = now;
            self.beat_env = 1.0;
            self.heart_v += LUB_IMPULSE; // the "lub"
            self.dub_at = now + 0.14; // the "dub" follows shortly after
            self.emit_ring(rect, 0.9);
            self.idle_at = now + 1.2; // hold off the resting pulse while music plays
        }
        // Resting heartbeat: a gentle auto-pulse when the music isn't beating.
        if now >= self.idle_at {
            self.idle_at = now + 0.95; // ~63 bpm at rest
            self.beat_env = self.beat_env.max(0.5);
            self.heart_v += LUB_IMPULSE * 0.5;
            self.dub_at = now + 0.14;
            self.emit_ring(rect, 0.45);
        }
        if now >= self.dub_at {
            self.dub_at = f32::MAX;
            self.heart_v += DUB_IMPULSE; // softer second thump (no extra ring)
        }

        // Underdamped spring (semi-implicit Euler): snap up, overshoot, settle.
        let accel = -HEART_K * self.heart_x - HEART_C * self.heart_v;
        self.heart_v += accel * dt;
        self.heart_x = (self.heart_x + self.heart_v * dt).clamp(-0.45, 0.7);

        self.beat_env *= (-dt * 5.5).exp();

        for r in &mut self.rings {
            r.age += dt;
        }
        self.rings.retain(|r| r.age < r.max_age);
    }

    fn emit_ring(&mut self, rect: Rect, strength: f32) {
        // `r0`/`speed` here are heart *widths*: the ring starts a touch larger
        // than the heart and grows outward.
        self.rings.push(PulseRing {
            age: 0.0,
            max_age: 0.55 + strength * 0.5,
            r0: rect.height() * 0.36,
            speed: rect.height() * (0.30 + strength * 0.25),
        });
        if self.rings.len() > 16 {
            self.rings.remove(0);
        }
    }

    /// Draw each pulse as an expanding, fading heart-shaped outline — a
    /// heartbeat radiating heartbeats.
    fn draw_rings(&self, painter: &egui::Painter, center: Pos2) {
        for r in &self.rings {
            let t = (r.age / r.max_age).clamp(0.0, 1.0);
            let size = r.r0 + r.speed * r.age;
            let alpha = (1.0 - t) * (1.0 - t) * 0.45;
            let width = 1.0 + (1.0 - t) * 2.0;
            let col = Color32::from_rgba_unmultiplied(0xff, 0x6f, 0xa6, (alpha * 255.0) as u8);
            let mut pts = heart_outline(center, size);
            pts.push(pts[0]);
            painter.add(egui::Shape::line(pts, egui::Stroke::new(width, col)));
        }
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        let mut cycle = false;
        let mut cycle_mode = false;
        let mut device_next = false;
        let mut device_prev = false;
        let mut toggle_overlay = false;
        let mut toggle_controls = false;
        let mut shoot = false;
        let mut quit = false;
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                cycle = true;
            }
            if i.key_pressed(egui::Key::M) {
                cycle_mode = true;
            }
            if i.key_pressed(egui::Key::D) {
                if i.modifiers.shift {
                    device_prev = true;
                } else {
                    device_next = true;
                }
            }
            if i.key_pressed(egui::Key::H) {
                toggle_overlay = true;
            }
            if i.key_pressed(egui::Key::Tab) {
                toggle_controls = true;
            }
            if i.key_pressed(egui::Key::S) {
                shoot = true;
            }
            if i.key_pressed(egui::Key::Q) || i.key_pressed(egui::Key::Escape) {
                quit = true;
            }
        });
        if cycle {
            self.palette_idx = (self.palette_idx + 1) % PALETTES.len();
        }
        if cycle_mode {
            self.mode_idx = (self.mode_idx + 1) % MODES.len();
        }
        if device_next {
            self.audio_control.next();
        }
        if device_prev {
            self.audio_control.prev();
        }
        if toggle_overlay {
            self.show_overlay = !self.show_overlay;
        }
        if toggle_controls {
            self.show_controls = !self.show_controls;
        }
        if shoot {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        if quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn save_pending_screenshots(&self, ctx: &egui::Context) {
        let images: Vec<std::sync::Arc<egui::ColorImage>> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
                .collect()
        });
        for image in images {
            let path = self
                .screenshot_path
                .clone()
                .unwrap_or_else(|| {
                    let n = self.screenshot_counter.fetch_add(1, Ordering::SeqCst);
                    std::path::PathBuf::from(format!("/tmp/facecam_{n:03}.png"))
                });
            if let Err(e) = save_color_image_png(&image, &path) {
                eprintln!("facecam: failed to save screenshot to {}: {e}", path.display());
            } else {
                eprintln!("facecam: screenshot → {}", path.display());
            }
        }
    }

    fn maybe_auto_screenshot(&mut self, ctx: &egui::Context, frame_count: u32) {
        if self.screenshot_path.is_none() {
            return;
        }
        if frame_count == 90 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        if frame_count >= 130 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

impl eframe::App for FacecamApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let dt = ui.ctx().input(|i| i.stable_dt).clamp(0.0, 0.1);
        if self.demo {
            self.feed_demo_signal(dt);
        } else {
            let n = self.consumer.pop_slice(&mut self.scratch);
            if n > 0 {
                self.analyzer.ingest(&self.scratch[..n]);
            }
        }
        self.analyzer.process();

        let track_snapshot = self.nowplaying.lock().unwrap().clone();
        if let (Some(prev), Some(curr)) = (&self.last_track, &track_snapshot) {
            if prev != curr {
                self.palette_idx = (self.palette_idx + 1) % PALETTES.len();
                eprintln!(
                    "facecam: track change → palette={}",
                    PALETTES[self.palette_idx].name
                );
            }
        }
        self.last_track = track_snapshot.clone();

        let ctx = ui.ctx().clone();
        self.handle_input(&ctx);

        let palette = song_palette_override(track_snapshot.as_ref())
            .unwrap_or(&PALETTES[self.palette_idx]);
        let mode = MODES[self.mode_idx];

        let rect = ui.max_rect();
        let painter = ui.painter().clone();

        // Track level + beats; drive the heartbeat spring and sonar rings.
        self.update_reactive(dt, rect);
        let time = self.start_time.elapsed().as_secs_f32();

        draw_background(&painter, rect, palette, mode, self.beat_env);

        let bar_count = self.analyzer.bars.len();
        let gap = 2.0;
        let bar_w =
            ((rect.width() - gap * (bar_count as f32 - 1.0)) / bar_count as f32).max(1.0);

        match mode {
            Mode::Full => {
                let max_h = rect.height() * 0.97;
                draw_bars(&painter, rect, &self.analyzer.bars, palette,
                          BarSide::Bottom, max_h, bar_w, gap);
            }
            Mode::Half => {
                let max_h = rect.height() * 0.5;
                draw_bars(&painter, rect, &self.analyzer.bars, palette,
                          BarSide::Bottom, max_h, bar_w, gap);
            }
            Mode::Mirror => {
                let max_h = rect.height() * 0.5;
                draw_bars(&painter, rect, &self.analyzer.bars, palette,
                          BarSide::Bottom, max_h, bar_w, gap);
                draw_bars(&painter, rect, &self.analyzer.bars, palette,
                          BarSide::Top, max_h, bar_w, gap);
            }
            Mode::Rainbow => {
                let max_h = rect.height() * 0.97;
                let phase = time * 1.50 + self.phase_offset;
                draw_rainbow_bars(&painter, rect, &self.analyzer.bars,
                                  max_h, bar_w, gap, phase);
            }
        }

        // The centerpiece: one beating heart with sonar rings radiating behind it.
        let heart_base = (rect.height() * 0.34).min(rect.width() * 0.20);
        let heart_center = Pos2::new(rect.center().x, rect.top() + heart_base * 1.05);
        let breathe = 0.025 * (time * 2.0).sin();
        let heart_size = heart_base * (1.0 + self.heart_x + self.level * 0.05 + breathe);
        // Velocity-driven squash & stretch: stretch tall on the upstroke, settle
        // wide on the recoil — the wobble of jelly.
        let squash = -self.heart_v * 0.03;
        self.draw_rings(&painter, heart_center);
        draw_heart(&painter, heart_center, heart_size, squash, self.beat_env, self.level);

        if self.show_overlay {
            let track = {
                let np = self.nowplaying.lock().unwrap();
                np.as_ref()
                    .map(|t| t.display())
                    .unwrap_or_else(|| String::from("(no track)"))
            };
            let palette_label = if matches!(mode, Mode::Rainbow) {
                format!("[{}]", mode.name())
            } else {
                format!("[{} | {}]", mode.name(), palette.name)
            };
            let device_label = self.audio_control.current().description;
            let font = egui::FontId::monospace(11.0);
            let pad = 3.0;

            let track_galley = painter.layout_no_wrap(track, font.clone(), Color32::WHITE);
            let track_pos =
                Pos2::new(rect.left() + 6.0, rect.bottom() - 4.0 - track_galley.size().y);
            let track_bg = Rect::from_min_size(
                Pos2::new(track_pos.x - pad, track_pos.y - pad),
                track_galley.size() + egui::vec2(pad * 2.0, pad * 2.0),
            );
            painter.rect_filled(track_bg, 0.0, Color32::BLACK);
            painter.galley(track_pos, track_galley, Color32::WHITE);

            let palette_galley = painter.layout_no_wrap(palette_label, font.clone(), Color32::WHITE);
            let palette_pos = Pos2::new(
                rect.right() - 6.0 - palette_galley.size().x,
                rect.bottom() - 4.0 - palette_galley.size().y,
            );
            let palette_bg = Rect::from_min_size(
                Pos2::new(palette_pos.x - pad, palette_pos.y - pad),
                palette_galley.size() + egui::vec2(pad * 2.0, pad * 2.0),
            );
            painter.rect_filled(palette_bg, 0.0, Color32::BLACK);
            painter.galley(palette_pos, palette_galley, Color32::WHITE);

            let device_galley = painter.layout_no_wrap(device_label, font, Color32::WHITE);
            let device_pos = Pos2::new(rect.left() + 6.0, rect.top() + 6.0);
            let device_bg = Rect::from_min_size(
                Pos2::new(device_pos.x - pad, device_pos.y - pad),
                device_galley.size() + egui::vec2(pad * 2.0, pad * 2.0),
            );
            painter.rect_filled(device_bg, 0.0, Color32::BLACK);
            painter.galley(device_pos, device_galley, Color32::WHITE);
        }

        if self.show_controls {
            draw_controls_panel(&painter, rect);
        }

        self.save_pending_screenshots(&ctx);
        let frame_count = ctx.cumulative_pass_nr() as u32;
        self.maybe_auto_screenshot(&ctx, frame_count);

        ctx.request_repaint();
    }
}

fn draw_bars(
    painter: &egui::Painter,
    rect: Rect,
    bars: &[f32],
    palette: &Palette,
    side: BarSide,
    max_h: f32,
    bar_w: f32,
    gap: f32,
) {
    let (anchor_y, direction) = match side {
        BarSide::Bottom => (rect.bottom(), -1.0_f32),
        BarSide::Top => (rect.top(), 1.0_f32),
    };

    let color_at_y = |y: f32| -> Color32 {
        let t = if max_h > 0.0 {
            (1.0 - direction * (y - anchor_y) / max_h).clamp(0.0, 1.0)
        } else {
            0.0
        };
        palette.sample(t)
    };

    for (i, &v) in bars.iter().enumerate() {
        let x0 = rect.left() + i as f32 * (bar_w + gap);
        let x1 = x0 + bar_w;
        let h = v * max_h;
        let tip_y = anchor_y + direction * h;
        let (y_min, y_max) = if direction < 0.0 {
            (tip_y, anchor_y)
        } else {
            (anchor_y, tip_y)
        };

        if h >= 0.5 {
            let mut row_ys: Vec<f32> = Vec::with_capacity(palette.stops.len() + 2);
            row_ys.push(y_min);
            let n_stops = palette.stops.len();
            if n_stops > 1 {
                for s in 0..n_stops {
                    let t = s as f32 / (n_stops - 1) as f32;
                    let stop_y = anchor_y + direction * (1.0 - t) * max_h;
                    if stop_y > y_min + 0.5 && stop_y < y_max - 0.5 {
                        row_ys.push(stop_y);
                    }
                }
            }
            row_ys.push(y_max);
            row_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut mesh = egui::Mesh::default();
            for &y in &row_ys {
                let color = color_at_y(y);
                mesh.vertices
                    .push(Vertex::untextured(Pos2::new(x0, y), color));
                mesh.vertices
                    .push(Vertex::untextured(Pos2::new(x1, y), color));
            }
            for k in 0..row_ys.len() - 1 {
                let tl = (k * 2) as u32;
                let tr = (k * 2 + 1) as u32;
                let bl = ((k + 1) * 2) as u32;
                let br = ((k + 1) * 2 + 1) as u32;
                mesh.indices.extend_from_slice(&[tl, bl, br, tl, br, tr]);
            }
            painter.add(egui::Shape::mesh(mesh));
        }

        // Bright bar tip + a soft bloom that intensifies with the bar.
        let cx = (x0 + x1) * 0.5;
        let tip_col = color_at_y(tip_y);
        if h >= 1.0 {
            soft_glow(
                painter,
                Pos2::new(cx, tip_y),
                (bar_w * 1.9).max(5.0) * (0.55 + v),
                (tip_col.r(), tip_col.g(), tip_col.b()),
                (v * 0.6).clamp(0.0, 0.7),
            );
        }
        let cap_h = 2.0;
        let cap_rect = if direction < 0.0 {
            Rect::from_min_max(
                Pos2::new(x0, (tip_y - cap_h).max(rect.top())),
                Pos2::new(x1, tip_y),
            )
        } else {
            Rect::from_min_max(
                Pos2::new(x0, tip_y),
                Pos2::new(x1, (tip_y + cap_h).min(rect.bottom())),
            )
        };
        painter.rect_filled(cap_rect, 0.0, lighten(tip_col, 0.55));
    }
}

fn draw_rainbow_bars(
    painter: &egui::Painter,
    rect: Rect,
    bars: &[f32],
    max_h: f32,
    bar_w: f32,
    gap: f32,
    phase: f32,
) {
    let anchor_y = rect.bottom();
    let n = bars.len().max(1);

    for (i, &v) in bars.iter().enumerate() {
        let hue = (i as f32 / n as f32 - phase).rem_euclid(1.0);
        let tip_color = hsv_to_color32(hue, 1.0, 1.0);
        let base_color = hsv_to_color32(hue, 1.0, 0.25);

        let x0 = rect.left() + i as f32 * (bar_w + gap);
        let x1 = x0 + bar_w;
        let h = v * max_h;
        let tip_y = anchor_y - h;

        if h >= 0.5 {
            let mut mesh = egui::Mesh::default();
            mesh.vertices
                .push(Vertex::untextured(Pos2::new(x0, tip_y), tip_color));
            mesh.vertices
                .push(Vertex::untextured(Pos2::new(x1, tip_y), tip_color));
            mesh.vertices
                .push(Vertex::untextured(Pos2::new(x0, anchor_y), base_color));
            mesh.vertices
                .push(Vertex::untextured(Pos2::new(x1, anchor_y), base_color));
            mesh.indices.extend_from_slice(&[0, 2, 3, 0, 3, 1]);
            painter.add(egui::Shape::mesh(mesh));
        }

        let cx = (x0 + x1) * 0.5;
        if h >= 1.0 {
            soft_glow(
                painter,
                Pos2::new(cx, tip_y),
                (bar_w * 1.9).max(5.0) * (0.55 + v),
                (tip_color.r(), tip_color.g(), tip_color.b()),
                (v * 0.6).clamp(0.0, 0.7),
            );
        }
        let cap_h = 2.0;
        let cap_rect = Rect::from_min_max(
            Pos2::new(x0, (tip_y - cap_h).max(rect.top())),
            Pos2::new(x1, tip_y),
        );
        painter.rect_filled(cap_rect, 0.0, lighten(tip_color, 0.55));
    }
}

fn hsv_to_color32(h: f32, s: f32, v: f32) -> Color32 {
    let h = h.rem_euclid(1.0) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color32::from_rgb(
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

/// Soft radial glow: a triangle fan from a translucent coloured centre out to a
/// fully transparent rim. Stacked over a dark background this reads as a bloom
/// without needing a custom additive blend pass.
fn soft_glow(painter: &egui::Painter, center: Pos2, radius: f32, rgb: (u8, u8, u8), alpha: f32) {
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha <= 0.004 || radius <= 0.5 {
        return;
    }
    const SEGMENTS: u32 = 24;
    let center_col = Color32::from_rgba_unmultiplied(rgb.0, rgb.1, rgb.2, (alpha * 255.0) as u8);
    let mut mesh = egui::Mesh::default();
    mesh.vertices.push(Vertex::untextured(center, center_col));
    for i in 0..=SEGMENTS {
        let ang = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
        let p = Pos2::new(center.x + ang.cos() * radius, center.y + ang.sin() * radius);
        mesh.vertices.push(Vertex::untextured(p, Color32::TRANSPARENT));
    }
    for i in 1..=SEGMENTS {
        mesh.indices.extend_from_slice(&[0, i, i + 1]);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// Four-corner quad with a top→bottom colour gradient.
fn gradient_quad_v(painter: &egui::Painter, rect: Rect, top: Color32, bottom: Color32) {
    let mut mesh = egui::Mesh::default();
    mesh.vertices.push(Vertex::untextured(rect.left_top(), top));
    mesh.vertices.push(Vertex::untextured(rect.right_top(), top));
    mesh.vertices.push(Vertex::untextured(rect.left_bottom(), bottom));
    mesh.vertices.push(Vertex::untextured(rect.right_bottom(), bottom));
    mesh.indices.extend_from_slice(&[0, 2, 3, 0, 3, 1]);
    painter.add(egui::Shape::mesh(mesh));
}

/// Background: a clean vertical gradient (dark top → palette base) and a subtle
/// beat flash. Deliberately spare — the heart and bars carry the detail.
fn draw_background(painter: &egui::Painter, rect: Rect, palette: &Palette, mode: Mode, beat: f32) {
    let base = if matches!(mode, Mode::Rainbow) {
        (0u8, 0u8, 0u8)
    } else {
        palette.bg
    };
    let dark = |c: u8, f: f32| (c as f32 * f) as u8;
    let top = Color32::from_rgb(dark(base.0, 0.30), dark(base.1, 0.30), dark(base.2, 0.30));
    let bottom = Color32::from_rgb(base.0, base.1, base.2);
    gradient_quad_v(painter, rect, top, bottom);

    if beat > 0.01 {
        let accent = palette.sample(0.0);
        let a = (beat * 0.08).clamp(0.0, 0.10);
        painter.rect_filled(
            rect,
            0.0,
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), (a * 255.0) as u8),
        );
    }
}

/// The centerpiece heart. Rather than a flat glyph it's a filled parametric
/// curve with a vertical gradient, a beat-driven glow, a glossy highlight, and a
/// bright rim — the one "simple" element rendered with the most care.
fn draw_heart(painter: &egui::Painter, center: Pos2, size: f32, squash: f32, beat: f32, level: f32) {
    let squash = squash.clamp(-0.14, 0.14);
    let pts = heart_outline_xy(center, size, 1.0 + squash, 1.0 - squash);

    // Outer glow that flares on the beat.
    let glow_a = (0.14 + level * 0.25 + beat * 0.45).clamp(0.0, 0.85);
    soft_glow(painter, center, size * 1.05, (0xff, 0x2b, 0x66), glow_a);

    // Vertical gradient fill (deep crimson at the point → hot pink at the lobes),
    // brightening as the beat hits.
    let lift = (beat * 0.5).clamp(0.0, 0.5);
    let top_rgb = lerp_rgb((0xff, 0x4f, 0x86), (0xff, 0x9f, 0xc4), lift);
    let bot_rgb = lerp_rgb((0xc8, 0x10, 0x3e), (0xff, 0x44, 0x77), lift);
    let (min_y, max_y) = pts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
        (lo.min(p.y), hi.max(p.y))
    });
    let span = (max_y - min_y).max(1.0);
    let color_at = |y: f32| {
        let t = ((y - min_y) / span).clamp(0.0, 1.0);
        let c = lerp_rgb(top_rgb, bot_rgb, t);
        Color32::from_rgb(c.0, c.1, c.2)
    };

    let mut mesh = egui::Mesh::default();
    mesh.vertices
        .push(Vertex::untextured(center, color_at(center.y)));
    for p in &pts {
        mesh.vertices.push(Vertex::untextured(*p, color_at(p.y)));
    }
    let rim = pts.len() as u32;
    for i in 1..=rim {
        let next = if i == rim { 1 } else { i + 1 };
        mesh.indices.extend_from_slice(&[0, i, next]);
    }
    painter.add(egui::Shape::mesh(mesh));

    // Soft inner shadow low in the belly for a rounded, inflated 3D look.
    soft_glow(
        painter,
        Pos2::new(center.x, center.y + size * 0.26),
        size * 0.40,
        (0x3a, 0x00, 0x12),
        0.28,
    );

    // Bright rim for a crisp neon edge; it flares toward white on the beat.
    let rim = lerp_rgb((0xff, 0xc6, 0xd8), (0xff, 0xff, 0xff), (beat * 0.8).clamp(0.0, 1.0));
    let mut loop_pts = pts.clone();
    loop_pts.push(pts[0]);
    painter.add(egui::Shape::line(
        loop_pts,
        egui::Stroke::new(1.5 + beat * 1.0, Color32::from_rgb(rim.0, rim.1, rim.2)),
    ));

    // Glossy specular highlight on the upper-left lobe, brighter on the beat,
    // with a small crisp catchlight on top for a wet, glassy sheen.
    soft_glow(
        painter,
        Pos2::new(center.x - size * 0.24, center.y - size * 0.2),
        size * 0.22,
        (0xff, 0xff, 0xff),
        (0.35 + level * 0.2 + beat * 0.25).clamp(0.0, 0.7),
    );
    soft_glow(
        painter,
        Pos2::new(center.x - size * 0.21, center.y - size * 0.22),
        size * 0.08,
        (0xff, 0xff, 0xff),
        0.85,
    );
}

/// Sample the classic heart curve `x = 16 sin³t`, `y = 13 cos t − 5 cos 2t −
/// 2 cos 3t − cos 4t`, mapped to screen space and centred on `center`.
fn heart_outline(center: Pos2, size: f32) -> Vec<Pos2> {
    heart_outline_xy(center, size, 1.0, 1.0)
}

/// As [`heart_outline`], but with independent x/y scale factors so the heart can
/// squash and stretch on the beat (`sx`>1, `sy`<1 = wider and shorter).
fn heart_outline_xy(center: Pos2, size: f32, sx: f32, sy: f32) -> Vec<Pos2> {
    const N: usize = 72;
    let scale = size / 32.0; // curve spans x ∈ [-16, 16]
    let mut raw: Vec<Pos2> = Vec::with_capacity(N);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    for i in 0..N {
        let t = i as f32 / N as f32 * std::f32::consts::TAU;
        let x = 16.0 * t.sin().powi(3);
        // Negate so the cusp is at the top in screen space (y grows downward).
        let y = -(13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos());
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        raw.push(Pos2::new(x, y));
    }
    let mid_y = (min_y + max_y) * 0.5;
    raw.into_iter()
        .map(|p| {
            Pos2::new(
                center.x + p.x * scale * sx,
                center.y + (p.y - mid_y) * scale * sy,
            )
        })
        .collect()
}

fn lerp_rgb(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let l = |x: u8, y: u8| ((1.0 - t) * x as f32 + t * y as f32) as u8;
    (l(a.0, b.0), l(a.1, b.1), l(a.2, b.2))
}

fn draw_controls_panel(painter: &egui::Painter, rect: Rect) {
    const ENTRIES: &[(&str, &str)] = &[
        ("Space", "cycle palette"),
        ("M", "cycle mode"),
        ("D / Shift+D", "next / prev audio device"),
        ("H", "toggle track overlay"),
        ("Tab", "toggle controls"),
        ("S", "screenshot"),
        ("Q / Esc", "quit"),
    ];

    let title_font = egui::FontId::proportional(18.0);
    let body_font = egui::FontId::monospace(14.0);
    let pad = 14.0;
    let line_h = 22.0;
    let title_gap = 12.0;
    let key_desc_gap = 16.0;

    let title_galley = painter.layout_no_wrap("Controls".to_string(), title_font.clone(), Color32::WHITE);
    let mut max_key_w = 0.0_f32;
    let mut max_desc_w = 0.0_f32;
    for (key, desc) in ENTRIES {
        let kg = painter.layout_no_wrap(key.to_string(), body_font.clone(), Color32::WHITE);
        let dg = painter.layout_no_wrap(desc.to_string(), body_font.clone(), Color32::WHITE);
        max_key_w = max_key_w.max(kg.size().x);
        max_desc_w = max_desc_w.max(dg.size().x);
    }
    let row_w = max_key_w + key_desc_gap + max_desc_w;
    let inner_w = row_w.max(title_galley.size().x);
    let panel_w = inner_w + pad * 2.0;
    let panel_h = pad * 2.0 + title_galley.size().y + title_gap + line_h * ENTRIES.len() as f32;

    let panel_rect = Rect::from_center_size(rect.center(), egui::vec2(panel_w, panel_h));
    painter.rect_filled(panel_rect, 0.0, Color32::from_rgba_premultiplied(0, 0, 0, 220));

    painter.text(
        Pos2::new(panel_rect.center().x, panel_rect.top() + pad),
        egui::Align2::CENTER_TOP,
        "Controls",
        title_font,
        Color32::WHITE,
    );

    let rows_left = panel_rect.center().x - row_w / 2.0;
    let rows_top = panel_rect.top() + pad + title_galley.size().y + title_gap;
    for (i, (key, desc)) in ENTRIES.iter().enumerate() {
        let y = rows_top + i as f32 * line_h;
        painter.text(
            Pos2::new(rows_left + max_key_w, y),
            egui::Align2::RIGHT_TOP,
            *key,
            body_font.clone(),
            Color32::from_rgb(255, 200, 200),
        );
        painter.text(
            Pos2::new(rows_left + max_key_w + key_desc_gap, y),
            egui::Align2::LEFT_TOP,
            *desc,
            body_font.clone(),
            Color32::WHITE,
        );
    }
}

fn song_palette_override(track: Option<&Track>) -> Option<&'static Palette> {
    let normalized: String = track?
        .title
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect();
    if normalized.contains("bumblebee") {
        return Some(&BUMBLEBEE);
    }
    None
}

fn lighten(c: Color32, amt: f32) -> Color32 {
    let amt = amt.clamp(0.0, 1.0);
    let mix = |v: u8| {
        let f = v as f32;
        (f + (255.0 - f) * amt).round().clamp(0.0, 255.0) as u8
    };
    Color32::from_rgba_premultiplied(mix(c.r()), mix(c.g()), mix(c.b()), c.a())
}

fn save_color_image_png(
    image: &egui::ColorImage,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::BufWriter;
    let [w, h] = image.size;
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let mut bytes = Vec::with_capacity(w * h * 4);
    for px in &image.pixels {
        bytes.extend_from_slice(&[px.r(), px.g(), px.b(), px.a()]);
    }
    writer.write_image_data(&bytes)?;
    Ok(())
}
