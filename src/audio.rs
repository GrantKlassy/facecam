use anyhow::{Context, Result, bail};
use ringbuf::{HeapRb, traits::{Producer, Split}};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub type AudioConsumer = ringbuf::HeapCons<f32>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioDevice {
    /// PulseAudio internal name (passed to `parec --device=`). Always a `.monitor`
    /// of an output sink — facecam never captures from microphones or other
    /// input sources, so the visualization tracks playback rather than the mic.
    pub name: String,
    /// Human-readable name shown in UI.
    pub description: String,
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
        bail!("no audio output monitors found via pactl");
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
        eprintln!("facecam: capturing from `{}`", device.description);

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
/// Input sources (microphones, line-in, etc.) are filtered out: facecam only
/// visualizes audio playback, so we capture exclusively from `.monitor` sources
/// of output sinks. Sorted alphabetically by description for stable cycling.
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

    devices.sort_by(|a, b| a.description.to_lowercase().cmp(&b.description.to_lowercase()));
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
        // Keep only `.monitor` sources. Prefer device.class when present;
        // fall back to the conventional `.monitor` name suffix when absent.
        let is_monitor = match self.device_class.as_deref() {
            Some("monitor") => true,
            Some(_) => false,
            None => name.ends_with(".monitor"),
        };
        if !is_monitor {
            return None;
        }
        Some(AudioDevice { name, description })
    }
}

/// Pick the initial device index. Substring match against name OR description
/// (case-insensitive). Falls back to the first device — which is always a
/// monitor since `parse_sources` filters out inputs.
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
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Realistic `pactl list sources` excerpt with a deliberate mix of inputs
    /// (HyperX mic, built-in analog) and `.monitor` sources of output sinks.
    /// Tests rely on this so the input/output filter is exercised by realistic
    /// pulseaudio output rather than synthetic edge cases alone.
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

    /// Test-only check that a parsed device is a monitor of an output sink.
    /// PulseAudio names every output monitor `<sink>.monitor`, so this suffix
    /// is a reliable post-filter assertion regardless of how `device.class`
    /// was reported.
    fn is_output_monitor(d: &AudioDevice) -> bool {
        d.name.ends_with(".monitor")
    }

    #[test]
    fn parses_real_pactl_output_excludes_inputs() {
        let devices = parse_sources(SAMPLE_PACTL);
        // Sample has 2 inputs and 2 monitors; only the monitors should remain.
        assert_eq!(devices.len(), 2);
        for d in &devices {
            assert!(
                is_output_monitor(d),
                "input source leaked into device list: {} ({})",
                d.name,
                d.description
            );
        }
        assert_eq!(
            devices[0].description,
            "Monitor of Built-in Audio Digital Stereo (IEC958)"
        );
        assert_eq!(devices[1].description, "Monitor of Record Player");
    }

    #[test]
    fn parse_sources_drops_input_with_class_sound() {
        let text = "Source #1\n\tName: alsa_input.usb-Mic\n\tDescription: My Mic\n\tProperties:\n\t\tdevice.class = \"sound\"\n";
        assert!(
            parse_sources(text).is_empty(),
            "input device with device.class=\"sound\" must be excluded on linux"
        );
    }

    #[test]
    fn parse_sources_drops_input_when_class_missing() {
        // No device.class — the parser falls back to checking the name suffix;
        // a non-`.monitor` name must be treated as input and dropped.
        let text = "Source #1\n\tName: alsa_input.usb-Mic\n\tDescription: My Mic\n";
        assert!(
            parse_sources(text).is_empty(),
            "input device without device.class must be excluded on linux"
        );
    }

    #[test]
    fn parse_sources_drops_unknown_class_values() {
        // Anything other than "monitor" is treated as a non-monitor source.
        let text = "Source #1\n\tName: weird.thing\n\tDescription: Weird\n\tProperties:\n\t\tdevice.class = \"abstract\"\n";
        assert!(parse_sources(text).is_empty());
    }

    #[test]
    fn classifies_monitor_via_name_suffix_when_class_missing() {
        let text = "Source #1\n\tName: foo.monitor\n\tDescription: Monitor of Foo\n";
        let devices = parse_sources(text);
        assert_eq!(devices.len(), 1);
        assert!(is_output_monitor(&devices[0]));
    }

    #[test]
    fn classifies_monitor_via_class_even_without_dot_monitor_name() {
        // Pathological but valid: device.class explicitly says monitor, so we
        // trust it even though the name doesn't follow convention. The test
        // helper deliberately won't recognize this — we assert presence and
        // length here instead.
        let text = "Source #1\n\tName: oddly_named\n\tDescription: Funky\n\tProperties:\n\t\tdevice.class = \"monitor\"\n";
        let devices = parse_sources(text);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "oddly_named");
    }

    #[test]
    fn skips_blocks_missing_required_fields() {
        let text = "Source #1\n\tDescription: Has no name\nSource #2\n\tName: ok.monitor\n\tDescription: ok\n";
        let devices = parse_sources(text);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "ok.monitor");
    }

    #[test]
    fn parses_empty_input() {
        assert!(parse_sources("").is_empty());
        assert!(parse_sources("garbage that doesn't look like pactl").is_empty());
    }

    #[test]
    fn pick_initial_never_returns_input_device() {
        // Sanity: after parse_sources filters inputs, pick_initial cannot
        // possibly hand back an input — none are present in the slice.
        let devices = parse_sources(SAMPLE_PACTL);
        assert!(!devices.is_empty());
        let i = pick_initial(&devices, None);
        assert!(is_output_monitor(&devices[i]));
    }

    #[test]
    fn pick_initial_with_input_substring_pref_still_returns_monitor() {
        // Even if FACECAM_DEVICE matches an input device's substring ("HyperX"),
        // there are no inputs in the device list — so we fall back to the first
        // monitor instead of accidentally selecting the mic.
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some("HyperX"));
        assert!(
            is_output_monitor(&devices[i]),
            "FACECAM_DEVICE matching an input substring must not select an input"
        );
    }

    #[test]
    fn pick_initial_matches_monitor_by_description_substring() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some("Record Player"));
        assert_eq!(devices[i].description, "Monitor of Record Player");
        assert!(is_output_monitor(&devices[i]));
    }

    #[test]
    fn pick_initial_is_case_insensitive() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some("record player"));
        assert_eq!(devices[i].description, "Monitor of Record Player");
    }

    #[test]
    fn pick_initial_matches_by_name_substring() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some("raop_sink"));
        assert_eq!(devices[i].description, "Monitor of Record Player");
        assert!(is_output_monitor(&devices[i]));
    }

    #[test]
    fn pick_initial_with_no_pref_picks_first_monitor() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, None);
        assert_eq!(i, 0);
        assert!(is_output_monitor(&devices[i]));
    }

    #[test]
    fn pick_initial_with_empty_pref_picks_first_monitor() {
        let devices = parse_sources(SAMPLE_PACTL);
        let i = pick_initial(&devices, Some(""));
        assert_eq!(i, 0);
        assert!(is_output_monitor(&devices[i]));
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

    fn dev(name: &str) -> AudioDevice {
        AudioDevice {
            name: name.to_string(),
            description: name.to_string(),
        }
    }

    #[test]
    fn control_next_wraps_around() {
        let ctrl = make_control(
            vec![dev("a.monitor"), dev("b.monitor"), dev("c.monitor")],
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
        let ctrl = make_control(vec![dev("a.monitor"), dev("b.monitor")], 0);
        ctrl.prev();
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(1));
    }

    #[test]
    fn control_next_sets_pending_not_selected() {
        // selected_idx must not change until the worker consumes the pending request,
        // so the overlay keeps showing the *active* device while parec restarts.
        let ctrl = make_control(vec![dev("a.monitor"), dev("b.monitor")], 0);
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
                dev("a.monitor"),
                dev("b.monitor"),
                dev("c.monitor"),
                dev("d.monitor"),
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
