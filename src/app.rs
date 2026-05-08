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
        let n = self.consumer.pop_slice(&mut self.scratch);
        if n > 0 {
            self.analyzer.ingest(&self.scratch[..n]);
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

        let rect = ui.max_rect();
        let painter = ui.painter().clone();

        let mode = MODES[self.mode_idx];
        let bg = if matches!(mode, Mode::Rainbow) {
            Color32::BLACK
        } else {
            palette.bg_color()
        };
        painter.rect_filled(rect, 0.0, bg);

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
                let phase = self.start_time.elapsed().as_secs_f32() * 1.50
                    + self.phase_offset;
                draw_rainbow_bars(&painter, rect, &self.analyzer.bars,
                                  max_h, bar_w, gap, phase);
            }
        }

        let face_text = "\u{2764}\u{2764}\u{2764}";
        let face_size = (rect.height() * 0.45).min(rect.width() / 6.0);
        painter.text(
            Pos2::new(rect.center().x, rect.top() + face_size * 0.05),
            egui::Align2::CENTER_TOP,
            face_text,
            egui::FontId::monospace(face_size),
            Color32::from_rgb(0xff, 0x33, 0x55),
        );

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
        let cap_color = lighten(color_at_y(tip_y), 0.45);
        painter.rect_filled(cap_rect, 0.0, cap_color);
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

        let cap_h = 2.0;
        let cap_rect = Rect::from_min_max(
            Pos2::new(x0, (tip_y - cap_h).max(rect.top())),
            Pos2::new(x1, tip_y),
        );
        painter.rect_filled(cap_rect, 0.0, lighten(tip_color, 0.45));
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
