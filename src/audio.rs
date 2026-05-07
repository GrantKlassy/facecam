use anyhow::{Context, Result, bail};
use ringbuf::{HeapRb, traits::{Producer, Split}};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub type AudioConsumer = ringbuf::HeapCons<f32>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// A physical (or virtual) recording source — microphone, line-in, etc.
    Input,
    /// A `.monitor` of an output sink — what the speakers/headphones are playing.
    OutputMonitor,
}

impl DeviceKind {
    pub fn label(&self) -> &'static str {
        match self {
            DeviceKind::Input => "in",
            DeviceKind::OutputMonitor => "out",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioDevice {
    /// PulseAudio internal name (passed to `parec --device=`).
    pub name: String,
    /// Human-readable name shown in UI.
    pub description: String,
    pub kind: DeviceKind,
}

pub struct AudioCapture {
    pub consumer: AudioConsumer,
    pub control: AudioControl,
}

#[derive(Clone)]
pub struct AudioControl {
    devices: Arc<Vec<AudioDevice>>,
    state: Arc<Mutex<ControlState>>,
}

struct ControlState {
    selected_idx: usize,
    pending_idx: Option<usize>,
}

impl AudioControl {
    pub fn selected_idx(&self) -> usize {
        self.state.lock().unwrap().selected_idx
    }

    pub fn current(&self) -> AudioDevice {
        self.devices[self.selected_idx()].clone()
    }

    pub fn next(&self) {
        self.advance(1);
    }

    pub fn prev(&self) {
        self.advance(-1);
    }

    /// Advance from the latest *intent*, not the active device — so if the user
    /// taps D twice before the worker has switched, they skip two forward instead
    /// of getting stuck on the same pending target.
    fn advance(&self, delta: isize) {
        let n = self.devices.len();
        if n == 0 {
            return;
        }
        let mut s = self.state.lock().unwrap();
        let from = s.pending_idx.unwrap_or(s.selected_idx) as isize;
        let next = (from + delta).rem_euclid(n as isize) as usize;
        s.pending_idx = Some(next);
    }
}

pub fn start(
    sample_rate: u32,
    ringbuf_capacity: usize,
    device_pref: Option<&str>,
) -> Result<AudioCapture> {
    let devices = list_devices()?;
    if devices.is_empty() {
        bail!("no audio sources found via pactl");
    }
    let initial_idx = pick_initial(&devices, device_pref);

    let devices = Arc::new(devices);
    let state = Arc::new(Mutex::new(ControlState {
        selected_idx: initial_idx,
        pending_idx: None,
    }));

    let rb = HeapRb::<f32>::new(ringbuf_capacity);
    let (prod, cons) = rb.split();

    let worker_devices = devices.clone();
    let worker_state = state.clone();
    thread::spawn(move || {
        capture_worker(worker_devices, worker_state, prod, sample_rate);
    });

    Ok(AudioCapture {
        consumer: cons,
        control: AudioControl { devices, state },
    })
}

fn capture_worker(
    devices: Arc<Vec<AudioDevice>>,
    state: Arc<Mutex<ControlState>>,
    mut producer: ringbuf::HeapProd<f32>,
    sample_rate: u32,
) {
    loop {
        let device = {
            let s = state.lock().unwrap();
            devices[s.selected_idx].clone()
        };
        eprintln!(
            "facecam: capturing from `{}` ({})",
            device.description,
            device.kind.label()
        );

        let child = Command::new("parec")
            .args([
                &format!("--device={}", device.name),
                "--format=float32le",
                &format!("--rate={sample_rate}"),
                "--channels=1",
                "--latency-msec=20",
                "--client-name=facecam",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                eprintln!("facecam: failed to spawn parec for {}: {e}", device.name);
                thread::sleep(Duration::from_millis(500));
                continue;
            }
        };

        let mut stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                let _ = child.kill();
                continue;
            }
        };

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
                    let _ = producer.push_slice(&scratch[..count]);
                }
                Err(_) => break,
            }

            let switch = {
                let mut s = state.lock().unwrap();
                if let Some(new_idx) = s.pending_idx.take() {
                    s.selected_idx = new_idx;
                    true
                } else {
                    false
                }
            };
            if switch {
                break;
            }
        }

        let _ = child.kill();
        let _ = child.wait();
    }
}

pub fn list_devices() -> Result<Vec<AudioDevice>> {
    let out = Command::new("pactl")
        .args(["list", "sources"])
        .output()
        .context("failed to run pactl — is pulseaudio-utils installed?")?;
    if !out.status.success() {
        bail!("pactl exited non-zero");
    }
    Ok(parse_sources(&String::from_utf8_lossy(&out.stdout)))
}

/// Parse `pactl list sources` output into a sorted device list.
///
/// Sort order: inputs first, then monitors. Within each group, alphabetical
/// by description. This is the order users will cycle through with the `D`
/// keybinding, so making it predictable matters.
pub fn parse_sources(text: &str) -> Vec<AudioDevice> {
    let mut devices = Vec::new();
    let mut current: Option<PartialDevice> = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Source #") {
            if let Some(d) = current.take().and_then(PartialDevice::finish) {
                devices.push(d);
            }
            current = Some(PartialDevice::default());
        } else if let Some(c) = current.as_mut() {
            if let Some(rest) = trimmed.strip_prefix("Name: ") {
                c.name = Some(rest.to_string());
            } else if let Some(rest) = trimmed.strip_prefix("Description: ") {
                c.description = Some(rest.to_string());
            } else if let Some(rest) = trimmed.strip_prefix("device.class = ") {
                c.device_class = Some(rest.trim_matches('"').to_string());
            }
        }
    }
    if let Some(d) = current.and_then(PartialDevice::finish) {
        devices.push(d);
    }

    devices.sort_by(|a, b| {
        // Inputs (0) come before monitors (1).
        let rank = |k: DeviceKind| match k {
            DeviceKind::Input => 0,
            DeviceKind::OutputMonitor => 1,
        };
        rank(a.kind)
            .cmp(&rank(b.kind))
            .then_with(|| a.description.to_lowercase().cmp(&b.description.to_lowercase()))
    });
    devices
}

#[derive(Default)]
struct PartialDevice {
    name: Option<String>,
    description: Option<String>,
    device_class: Option<String>,
}

impl PartialDevice {
    fn finish(self) -> Option<AudioDevice> {
        let name = self.name?;
        let description = self.description?;
        // Prefer device.class when present; fall back to .monitor name suffix.
        let kind = match self.device_class.as_deref() {
            Some("monitor") => DeviceKind::OutputMonitor,
            Some(_) => DeviceKind::Input,
            None => {
                if name.ends_with(".monitor") {
                    DeviceKind::OutputMonitor
                } else {
                    DeviceKind::Input
                }
            }
        };
        Some(AudioDevice {
            name,
            description,
            kind,
        })
    }
}

/// Pick the initial device index. Substring match against name OR description
/// (case-insensitive). Falls back to the first input, then to index 0.
pub fn pick_initial(devices: &[AudioDevice], pref: Option<&str>) -> usize {
    if let Some(needle) = pref {
        let lower = needle.to_lowercase();
        if !lower.is_empty() {
            if let Some(i) = devices.iter().position(|d| {
                d.name.to_lowercase().contains(&lower)
                    || d.description.to_lowercase().contains(&lower)
            }) {
                return i;
            }
        }
    }
    devices
        .iter()
        .position(|d| d.kind == DeviceKind::Input)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PACTL: &str = r#"Source #57
	State: SUSPENDED
	Name: raop_sink.Sonos-38420B8EDBB8.local.10.0.0.104.7000.monitor
	Description: Monitor of Record Player
	Driver: PipeWire
	Properties:
		device.class = "monitor"
Source #73
	State: SUSPENDED
	Name: alsa_input.usb-Kingston_HyperX_SoloCast-00.analog-stereo
	Description: HyperX SoloCast Analog Stereo
	Driver: PipeWire
	Properties:
		device.class = "sound"
Source #74
	State: SUSPENDED
	Name: alsa_output.pci-0000_00_1f.3.iec958-stereo.monitor
	Description: Monitor of Built-in Audio Digital Stereo (IEC958)
	Driver: PipeWire
	Properties:
		device.class = "monitor"
Source #75
	State: SUSPENDED
	Name: alsa_input.pci-0000_00_1f.3.analog-stereo
	Description: Built-in Audio Analog Stereo
	Driver: PipeWire
	Properties:
		device.class = "sound"
"#;

    #[test]
    fn parses_real_pactl_output() {
        let devices = parse_sources(SAMPLE_PACTL);
        assert_eq!(devices.len(), 4);

        // Inputs first (sorted alphabetically by description).
        assert_eq!(devices[0].description, "Built-in Audio Analog Stereo");
        assert_eq!(devices[0].kind, DeviceKind::Input);
        assert_eq!(devices[1].description, "HyperX SoloCast Analog Stereo");
        assert_eq!(devices[1].kind, DeviceKind::Input);
        assert_eq!(
            devices[1].name,
            "alsa_input.usb-Kingston_HyperX_SoloCast-00.analog-stereo"
        );

        // Then monitors.
        assert_eq!(devices[2].kind, DeviceKind::OutputMonitor);
        assert_eq!(devices[3].kind, DeviceKind::OutputMonitor);
        assert!(devices[2].description.starts_with("Monitor of"));
    }

    #[test]
    fn classifies_monitor_via_name_suffix_when_class_missing() {
        let text = "Source #1\n\tName: foo.monitor\n\tDescription: Monitor of Foo\n";
        let devices = parse_sources(text);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].kind, DeviceKind::OutputMonitor);
    }

    #[test]
    fn skips_blocks_missing_required_fields() {
        let text = "Source #1\n\tDescription: Has no name\nSource #2\n\tName: ok\n\tDescription: ok\n";
        let devices = parse_sources(text);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "ok");
    }

    #[test]
    fn parses_empty_input() {
        assert!(parse_sources("").is_empty());
        assert!(parse_sources("garbage that doesn't look like pactl").is_empty());
    }

    #[test]
    fn pick_initial_matches_by_description_substring() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some("HyperX"));
        assert_eq!(devices[i].description, "HyperX SoloCast Analog Stereo");
    }

    #[test]
    fn pick_initial_is_case_insensitive() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some("hyperx"));
        assert_eq!(devices[i].description, "HyperX SoloCast Analog Stereo");
    }

    #[test]
    fn pick_initial_matches_by_name_substring() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some("Kingston"));
        assert_eq!(devices[i].description, "HyperX SoloCast Analog Stereo");
    }

    #[test]
    fn pick_initial_falls_back_to_first_input_when_no_match() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some("nonexistent device"));
        assert_eq!(devices[i].kind, DeviceKind::Input);
        assert_eq!(devices[i].description, "Built-in Audio Analog Stereo");
    }

    #[test]
    fn pick_initial_with_no_pref_picks_first_input() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, None);
        assert_eq!(devices[i].kind, DeviceKind::Input);
    }

    #[test]
    fn pick_initial_with_empty_pref_picks_first_input() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some(""));
        assert_eq!(devices[i].kind, DeviceKind::Input);
    }

    #[test]
    fn pick_initial_with_only_monitors_picks_zero() {
        let text = "Source #1\n\tName: foo.monitor\n\tDescription: Foo\n";
        let devices = parse_sources(text);
        assert_eq!(pick_initial(&devices, None), 0);
    }

    fn make_control(devices: Vec<AudioDevice>, initial: usize) -> AudioControl {
        AudioControl {
            devices: Arc::new(devices),
            state: Arc::new(Mutex::new(ControlState {
                selected_idx: initial,
                pending_idx: None,
            })),
        }
    }

    fn dev(name: &str, kind: DeviceKind) -> AudioDevice {
        AudioDevice {
            name: name.to_string(),
            description: name.to_string(),
            kind,
        }
    }

    #[test]
    fn control_next_wraps_around() {
        let ctrl = make_control(
            vec![
                dev("a", DeviceKind::Input),
                dev("b", DeviceKind::Input),
                dev("c", DeviceKind::OutputMonitor),
            ],
            0,
        );
        assert_eq!(ctrl.selected_idx(), 0);
        ctrl.next();
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(1));
        // Simulate the worker consuming the request:
        let mut s = ctrl.state.lock().unwrap();
        s.selected_idx = s.pending_idx.take().unwrap();
        drop(s);
        assert_eq!(ctrl.selected_idx(), 1);

        ctrl.next();
        ctrl.next(); // would go past end
        let pending = ctrl.state.lock().unwrap().pending_idx;
        assert_eq!(pending, Some(0)); // wrapped
    }

    #[test]
    fn control_prev_wraps_around() {
        let ctrl = make_control(
            vec![dev("a", DeviceKind::Input), dev("b", DeviceKind::Input)],
            0,
        );
        ctrl.prev();
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(1));
    }

    #[test]
    fn control_next_sets_pending_not_selected() {
        // selected_idx must not change until the worker consumes the pending request,
        // so the overlay keeps showing the *active* device while parec restarts.
        let ctrl = make_control(
            vec![dev("a", DeviceKind::Input), dev("b", DeviceKind::Input)],
            0,
        );
        ctrl.next();
        assert_eq!(ctrl.selected_idx(), 0);
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(1));
    }

    #[test]
    fn control_rapid_taps_skip_forward() {
        // Tapping next twice before the worker has consumed the first request
        // should advance two steps, not stay on +1.
        let ctrl = make_control(
            vec![
                dev("a", DeviceKind::Input),
                dev("b", DeviceKind::Input),
                dev("c", DeviceKind::Input),
                dev("d", DeviceKind::Input),
            ],
            0,
        );
        ctrl.next();
        ctrl.next();
        ctrl.next();
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(3));
    }

    #[test]
    fn control_next_on_empty_is_noop() {
        let ctrl = make_control(vec![], 0);
        ctrl.next();
        ctrl.prev();
        // Should not panic.
    }
}
