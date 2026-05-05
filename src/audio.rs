use anyhow::{Context, Result, bail};
use ringbuf::{HeapRb, traits::{Producer, Split}};
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;

const TARGET_DESCRIPTION: &str = "Built-in Audio Optical";

pub type AudioConsumer = ringbuf::HeapCons<f32>;

pub struct AudioCapture {
    pub consumer: AudioConsumer,
    pub source_name: String,
}

pub fn start(sample_rate: u32, ringbuf_capacity: usize) -> Result<AudioCapture> {
    let source_name = find_source()
        .context("could not find an audio source matching 'Built-in Audio Optical'")?;

    let rb = HeapRb::<f32>::new(ringbuf_capacity);
    let (mut prod, cons) = rb.split();

    let mut child = Command::new("parec")
        .args([
            &format!("--device={source_name}"),
            "--format=float32le",
            &format!("--rate={sample_rate}"),
            "--channels=1",
            "--latency-msec=20",
            "--client-name=facecam",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn parec — is pulseaudio-utils installed?")?;

    let mut stdout = child.stdout.take().context("parec stdout missing")?;

    thread::spawn(move || {
        let mut buf = [0u8; 4096];
        let mut scratch = [0f32; 1024];
        loop {
            match stdout.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let aligned = n - (n % 4);
                    let count = aligned / 4;
                    for i in 0..count {
                        let off = i * 4;
                        scratch[i] = f32::from_le_bytes([
                            buf[off], buf[off + 1], buf[off + 2], buf[off + 3],
                        ]);
                    }
                    let _ = prod.push_slice(&scratch[..count]);
                }
                Err(_) => break,
            }
        }
        let _ = child.kill();
    });

    Ok(AudioCapture { consumer: cons, source_name })
}

fn find_source() -> Result<String> {
    let out = Command::new("pactl")
        .args(["list", "sources"])
        .output()
        .context("failed to run pactl — is pulseaudio-utils installed?")?;
    if !out.status.success() {
        bail!("pactl exited non-zero");
    }
    let text = String::from_utf8_lossy(&out.stdout);

    let mut current_name: Option<String> = None;
    let mut matched: Option<String> = None;
    let mut first_monitor: Option<String> = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Source #") {
            current_name = None;
        } else if let Some(rest) = trimmed.strip_prefix("Name: ") {
            current_name = Some(rest.to_string());
            if first_monitor.is_none() && rest.ends_with(".monitor") {
                first_monitor = Some(rest.to_string());
            }
        } else if let Some(rest) = trimmed.strip_prefix("Description: ") {
            if rest.contains(TARGET_DESCRIPTION) {
                if let Some(name) = current_name.take() {
                    matched = Some(name);
                    break;
                }
            }
        }
    }

    matched
        .or(first_monitor)
        .context("no source matched 'Built-in Audio Optical' and no .monitor source available")
}
