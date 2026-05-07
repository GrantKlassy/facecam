use anyhow::{Context, Result};
use eframe::egui;
use std::sync::Arc;

mod app;
mod audio;
mod colors;
mod fft;
mod nowplaying;

fn main() -> Result<()> {
    let device_pref = std::env::var("FACECAM_DEVICE").ok();
    let capture = audio::start(app::SAMPLE_RATE, 32_768, device_pref.as_deref())
        .context("failed to start audio capture")?;

    let nowplaying = nowplaying::start();

    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 520.0])
            .with_min_inner_size([320.0, 120.0])
            .with_title("facecam"),
        ..Default::default()
    };

    eframe::run_native(
        "facecam",
        opts,
        Box::new(move |cc| {
            install_cjk_fallback(&cc.egui_ctx);
            Ok(Box::new(app::FacecamApp::new(
                capture.consumer,
                capture.control,
                nowplaying,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;
    Ok(())
}

fn install_cjk_fallback(ctx: &egui::Context) {
    const CJK_CANDIDATES: &[(&str, u32)] = &[
        ("/usr/share/fonts/google-noto-sans-mono-cjk-vf-fonts/NotoSansMonoCJK-VF.ttc", 0),
        ("/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc", 0),
        ("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf", 0),
    ];
    const EMOJI_CANDIDATES: &[(&str, u32)] = &[
        ("/usr/share/fonts/google-noto-color-emoji-fonts/Noto-COLRv1.ttf", 0),
        ("/usr/share/fonts/google-noto-emoji-fonts/NotoEmoji-Regular.ttf", 0),
    ];

    let mut fonts = egui::FontDefinitions::default();
    let mut installed: Vec<&str> = Vec::new();

    for (key, candidates) in [
        ("cjk_fallback", CJK_CANDIDATES),
        ("emoji_fallback", EMOJI_CANDIDATES),
    ] {
        let Some((bytes, index)) = candidates.iter().find_map(|(path, idx)| {
            std::fs::read(path).ok().map(|b| (b, *idx))
        }) else {
            continue;
        };
        let mut data = egui::FontData::from_owned(bytes);
        data.index = index;
        fonts.font_data.insert(key.to_owned(), Arc::new(data));
        installed.push(key);
    }

    if installed.is_empty() {
        eprintln!("facecam: no CJK/emoji font found; non-ASCII glyphs may render as boxes");
        return;
    }

    for key in installed {
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .push(key.to_owned());
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .push(key.to_owned());
    }
    ctx.set_fonts(fonts);
}
