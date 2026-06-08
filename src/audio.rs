use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, Host, Sample, SampleFormat, SizedSample, StreamConfig};
use ringbuf::{
    HeapRb,
    traits::{Producer, Split},
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub type AudioConsumer = ringbuf::HeapCons<f32>;
type SharedProducer = Arc<Mutex<ringbuf::HeapProd<f32>>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioDevice {
    /// CoreAudio device name as reported by cpal. Used both to re-open the device
    /// for capture and as the human-readable label in the UI.
    pub name: String,
    /// Human-readable name shown in UI (identical to `name` on macOS/cpal).
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
        bail!("no audio input devices found via cpal");
    }
    let initial_idx = pick_initial(&devices, device_pref);

    let devices = Arc::new(devices);
    let state = Arc::new(Mutex::new(ControlState {
        selected_idx: initial_idx,
        pending_idx: None,
    }));

    let rb = HeapRb::<f32>::new(ringbuf_capacity);
    let (prod, cons) = rb.split();
    let producer: SharedProducer = Arc::new(Mutex::new(prod));

    let worker_devices = devices.clone();
    let worker_state = state.clone();
    thread::spawn(move || {
        capture_worker(worker_devices, worker_state, producer, sample_rate);
    });

    Ok(AudioCapture {
        consumer: cons,
        control: AudioControl { devices, state },
    })
}

/// Whether the capture stream stopped because the user switched devices or
/// because the stream ended on its own (error / device disappeared).
enum StreamExit {
    Switch,
    Ended,
}

fn capture_worker(
    devices: Arc<Vec<AudioDevice>>,
    state: Arc<Mutex<ControlState>>,
    producer: SharedProducer,
    target_rate: u32,
) {
    loop {
        let device = {
            let s = state.lock().unwrap();
            devices[s.selected_idx].clone()
        };
        eprintln!("facecam: capturing from `{}`", device.description);

        match run_stream(&device, &producer, target_rate, &state) {
            Ok(StreamExit::Switch) => continue,
            Ok(StreamExit::Ended) => thread::sleep(Duration::from_millis(500)),
            Err(e) => {
                eprintln!("facecam: capture error for {}: {e:#}", device.name);
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

/// Open `info` as a cpal input stream and pump mono f32 samples into the ringbuf
/// until the user requests a different device or the stream dies. The cpal
/// `Stream` is created, played, and dropped entirely on the calling thread (it is
/// `!Send`), so it never crosses a thread boundary.
fn run_stream(
    info: &AudioDevice,
    producer: &SharedProducer,
    target_rate: u32,
    state: &Arc<Mutex<ControlState>>,
) -> Result<StreamExit> {
    let host = cpal::default_host();
    let device =
        find_device_by_name(&host, &info.name).ok_or_else(|| anyhow!("device `{}` not found", info.name))?;

    let supported = choose_config(&device, target_rate)?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.config();
    let channels = config.channels.max(1) as usize;
    let capture_rate = config.sample_rate;

    let dead = Arc::new(AtomicBool::new(false));

    let stream = match sample_format {
        SampleFormat::F32 => {
            build_input::<f32>(&device, &config, channels, capture_rate, target_rate, producer.clone(), dead.clone())
        }
        SampleFormat::I16 => {
            build_input::<i16>(&device, &config, channels, capture_rate, target_rate, producer.clone(), dead.clone())
        }
        SampleFormat::U16 => {
            build_input::<u16>(&device, &config, channels, capture_rate, target_rate, producer.clone(), dead.clone())
        }
        other => bail!("unsupported sample format: {other:?}"),
    }?;

    stream.play().context("failed to start capture stream")?;

    loop {
        thread::sleep(Duration::from_millis(50));
        if dead.load(Ordering::SeqCst) {
            return Ok(StreamExit::Ended);
        }
        let switch = {
            let mut s = state.lock().unwrap();
            if let Some(idx) = s.pending_idx.take() {
                s.selected_idx = idx;
                true
            } else {
                false
            }
        };
        if switch {
            return Ok(StreamExit::Switch);
        }
    }
}

fn build_input<T>(
    device: &Device,
    config: &StreamConfig,
    channels: usize,
    capture_rate: u32,
    target_rate: u32,
    producer: SharedProducer,
    dead: Arc<AtomicBool>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let err_dead = dead.clone();
    let stream = device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let mut mono: Vec<f32> = Vec::with_capacity(data.len() / channels + 1);
                for frame in data.chunks(channels) {
                    let mut sum = 0.0f32;
                    for &s in frame {
                        sum += f32::from_sample(s);
                    }
                    mono.push(sum / channels as f32);
                }
                let out = resample_linear(&mono, capture_rate, target_rate);
                if let Ok(mut p) = producer.lock() {
                    let _ = p.push_slice(&out);
                }
            },
            move |e| {
                eprintln!("facecam: stream error: {e}");
                err_dead.store(true, Ordering::SeqCst);
            },
            None,
        )
        .context("failed to build input stream")?;
    Ok(stream)
}

/// Per-chunk linear resample from `from` to `to` Hz. Stateless across chunks: the
/// tiny discontinuity at chunk boundaries is inaudible and invisible for a
/// spectrum visualizer, and the common path (BlackHole at 44.1 kHz) is a no-op.
fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to as f32 / from as f32;
    let out_len = ((input.len() as f32) * ratio).round().max(1.0) as usize;
    let last = input.len() - 1;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f32 / ratio;
        let idx = src.floor() as usize;
        let frac = src - idx as f32;
        let a = input[idx.min(last)];
        let b = input[(idx + 1).min(last)];
        out.push(a + (b - a) * frac);
    }
    out
}

#[allow(deprecated)]
fn find_device_by_name(host: &Host, name: &str) -> Option<Device> {
    host.input_devices()
        .ok()?
        .find(|d| d.name().map(|n| n == name).unwrap_or(false))
}

/// Pick a stream config for `device`, preferring one that supports `target_rate`
/// natively with an f32 sample format and the fewest channels. Falls back to the
/// device's default config (and we resample) when the target rate isn't offered.
fn choose_config(device: &Device, target_rate: u32) -> Result<cpal::SupportedStreamConfig> {
    let target = target_rate;
    let mut best: Option<cpal::SupportedStreamConfigRange> = None;
    if let Ok(ranges) = device.supported_input_configs() {
        for r in ranges {
            if r.min_sample_rate() <= target && target <= r.max_sample_rate() {
                let better = match &best {
                    None => true,
                    Some(b) => {
                        let r_f32 = r.sample_format() == SampleFormat::F32;
                        let b_f32 = b.sample_format() == SampleFormat::F32;
                        if r_f32 != b_f32 {
                            r_f32
                        } else {
                            r.channels() < b.channels()
                        }
                    }
                };
                if better {
                    best = Some(r);
                }
            }
        }
    }
    if let Some(r) = best {
        return Ok(r.with_sample_rate(target));
    }
    device
        .default_input_config()
        .context("device has no default input config")
}

#[allow(deprecated)]
pub fn list_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .context("failed to enumerate input devices")?;
    let mut out = Vec::new();
    for d in devices {
        if let Ok(name) = d.name() {
            out.push(AudioDevice {
                name: name.clone(),
                description: name,
            });
        }
    }
    out.sort_by(|a, b| a.description.to_lowercase().cmp(&b.description.to_lowercase()));
    out.dedup_by(|a, b| a.name == b.name);
    Ok(out)
}

/// Pick the initial device index. With an explicit preference (`FACECAM_DEVICE`),
/// substring-match it against name/description (case-insensitive). With no
/// preference, prefer a BlackHole loopback so the visualizer tracks system
/// playback out of the box; otherwise fall back to the first device.
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
    if let Some(i) = devices
        .iter()
        .position(|d| d.description.to_lowercase().contains("blackhole"))
    {
        return i;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(name: &str) -> AudioDevice {
        AudioDevice {
            name: name.to_string(),
            description: name.to_string(),
        }
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

    #[test]
    fn pick_initial_matches_by_substring() {
        let devices = vec![dev("MacBook Pro Microphone"), dev("BlackHole 2ch"), dev("USB Audio")];
        assert_eq!(pick_initial(&devices, Some("usb")), 2);
        assert_eq!(pick_initial(&devices, Some("Microphone")), 0);
    }

    #[test]
    fn pick_initial_is_case_insensitive() {
        let devices = vec![dev("MacBook Pro Microphone"), dev("BlackHole 2ch")];
        assert_eq!(pick_initial(&devices, Some("blackhole 2ch")), 1);
    }

    #[test]
    fn pick_initial_prefers_blackhole_when_no_pref() {
        // Order is arbitrary; the loopback must win regardless of position.
        let devices = vec![dev("MacBook Pro Microphone"), dev("BlackHole 16ch"), dev("USB Audio")];
        assert_eq!(pick_initial(&devices, None), 1);
    }

    #[test]
    fn pick_initial_falls_back_to_first_without_blackhole() {
        let devices = vec![dev("MacBook Pro Microphone"), dev("USB Audio")];
        assert_eq!(pick_initial(&devices, None), 0);
        assert_eq!(pick_initial(&devices, Some("")), 0);
    }

    #[test]
    fn pick_initial_unmatched_pref_falls_back() {
        // A preference that matches nothing falls through to the BlackHole rule.
        let devices = vec![dev("MacBook Pro Microphone"), dev("BlackHole 2ch")];
        assert_eq!(pick_initial(&devices, Some("nonexistent")), 1);
    }

    #[test]
    fn resample_noop_when_rates_match() {
        let input = [0.0, 0.5, -0.5, 1.0];
        assert_eq!(resample_linear(&input, 44100, 44100), input);
    }

    #[test]
    fn resample_downsamples_length() {
        let input: Vec<f32> = (0..480).map(|i| i as f32).collect();
        let out = resample_linear(&input, 48000, 44100);
        assert_eq!(out.len(), 441);
        // Endpoints are preserved within interpolation error.
        assert!((out[0] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn control_next_wraps_around() {
        let ctrl = make_control(vec![dev("a"), dev("b"), dev("c")], 0);
        assert_eq!(ctrl.selected_idx(), 0);
        ctrl.next();
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(1));
        // Simulate the worker consuming the request:
        {
            let mut s = ctrl.state.lock().unwrap();
            s.selected_idx = s.pending_idx.take().unwrap();
        }
        assert_eq!(ctrl.selected_idx(), 1);
        ctrl.next();
        ctrl.next(); // would go past end
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(0)); // wrapped
    }

    #[test]
    fn control_prev_wraps_around() {
        let ctrl = make_control(vec![dev("a"), dev("b")], 0);
        ctrl.prev();
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(1));
    }

    #[test]
    fn control_next_sets_pending_not_selected() {
        // selected_idx must not change until the worker consumes the pending request,
        // so the overlay keeps showing the *active* device while the stream restarts.
        let ctrl = make_control(vec![dev("a"), dev("b")], 0);
        ctrl.next();
        assert_eq!(ctrl.selected_idx(), 0);
        assert_eq!(ctrl.state.lock().unwrap().pending_idx, Some(1));
    }

    #[test]
    fn control_rapid_taps_skip_forward() {
        let ctrl = make_control(vec![dev("a"), dev("b"), dev("c"), dev("d")], 0);
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
