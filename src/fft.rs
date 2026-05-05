use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub struct Analyzer {
    fft: Arc<dyn Fft<f32>>,
    fft_size: usize,
    window: Vec<f32>,
    scratch: Vec<Complex<f32>>,
    work: Vec<Complex<f32>>,
    samples: Vec<f32>,
    write_pos: usize,
    bar_bins: Vec<(usize, usize)>,
    pub bars: Vec<f32>,
    raw_bars: Vec<f32>,
    attack: f32,
    decay: f32,
}

impl Analyzer {
    pub fn new(
        fft_size: usize,
        sample_rate: u32,
        num_bars: usize,
        low_hz: f32,
        high_hz: f32,
    ) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

        let window: Vec<f32> = (0..fft_size)
            .map(|n| {
                0.5 - 0.5
                    * (2.0 * std::f32::consts::PI * n as f32 / (fft_size - 1) as f32).cos()
            })
            .collect();

        let bin_hz = sample_rate as f32 / fft_size as f32;
        let half = fft_size / 2;
        let log_low = low_hz.ln();
        let log_high = high_hz.ln();
        let first_log_bin = ((low_hz / bin_hz) as usize).max(1);
        let mut bar_bins = Vec::with_capacity(num_bars + 2);
        if first_log_bin > 1 {
            bar_bins.push((1, first_log_bin));
        }
        let mut prev_bin = first_log_bin;
        for i in 0..num_bars {
            let t = (i + 1) as f32 / num_bars as f32;
            let hz = (log_low + (log_high - log_low) * t).exp();
            let mut end_bin = (hz / bin_hz) as usize;
            if end_bin <= prev_bin {
                end_bin = prev_bin + 1;
            }
            if end_bin > half {
                end_bin = half;
            }
            bar_bins.push((prev_bin, end_bin));
            prev_bin = end_bin;
        }
        if prev_bin < half {
            bar_bins.push((prev_bin, half));
        }
        let total_bars = bar_bins.len();

        Self {
            fft,
            fft_size,
            window,
            scratch,
            work: vec![Complex::new(0.0, 0.0); fft_size],
            samples: vec![0.0; fft_size],
            write_pos: 0,
            bar_bins,
            bars: vec![0.0; total_bars],
            raw_bars: vec![0.0; total_bars],
            attack: 0.65,
            decay: 0.06,
        }
    }

    pub fn ingest(&mut self, new_samples: &[f32]) {
        for &s in new_samples {
            self.samples[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.fft_size;
        }
    }

    pub fn process(&mut self) {
        for i in 0..self.fft_size {
            let src = (self.write_pos + i) % self.fft_size;
            self.work[i].re = self.samples[src] * self.window[i];
            self.work[i].im = 0.0;
        }
        self.fft
            .process_with_scratch(&mut self.work, &mut self.scratch);

        let half = self.fft_size / 2;
        // Hann window has coherent gain of 0.5, so 4/N (not 2/N) recovers true amplitude
        let norm = 4.0 / self.fft_size as f32;
        for (i, &(a, b)) in self.bar_bins.iter().enumerate() {
            let a = a.min(half);
            let b = b.min(half).max(a + 1);
            let mut peak: f32 = 0.0;
            for k in a..b {
                let m =
                    (self.work[k].re * self.work[k].re + self.work[k].im * self.work[k].im).sqrt()
                        * norm;
                if m > peak {
                    peak = m;
                }
            }
            let db = 20.0 * peak.max(1e-7).log10();
            // Pre-emphasis: music's spectrum naturally rolls off ~3 dB/octave above ~500 Hz;
            // boost each bar going right so treble bars feel as alive as bass.
            let preemphasis_db = i as f32 * 0.30;
            let v = ((db + preemphasis_db + 55.0) / 50.0).clamp(0.0, 1.0);
            self.raw_bars[i] = v;
        }

        for (smooth, &raw) in self.bars.iter_mut().zip(self.raw_bars.iter()) {
            if raw > *smooth {
                *smooth += (raw - *smooth) * self.attack;
            } else {
                *smooth += (raw - *smooth) * self.decay;
            }
        }
    }
}
