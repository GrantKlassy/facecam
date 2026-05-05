use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Track {
    pub title: String,
    pub artist: String,
}

impl Track {
    pub fn display(&self) -> String {
        match (self.artist.is_empty(), self.title.is_empty()) {
            (true, true) => String::from("(no track)"),
            (true, false) => self.title.clone(),
            (false, true) => self.artist.clone(),
            (false, false) => format!("{} — {}", self.artist, self.title),
        }
    }
}

pub type SharedTrack = Arc<Mutex<Option<Track>>>;

pub fn start() -> SharedTrack {
    let state: SharedTrack = Arc::new(Mutex::new(None));
    let writer = state.clone();
    thread::spawn(move || loop {
        let next = poll_once();
        if let Some(t) = next {
            let mut guard = writer.lock().unwrap();
            if guard.as_ref() != Some(&t) {
                *guard = Some(t);
            }
        }
        thread::sleep(Duration::from_millis(800));
    });
    state
}

fn poll_once() -> Option<Track> {
    let finder = mpris::PlayerFinder::new().ok()?;
    let player = finder.find_active().ok()?;
    let meta = player.get_metadata().ok()?;
    let title = meta.title().unwrap_or("").to_string();
    let artist = meta
        .artists()
        .and_then(|a| a.first().map(|s| s.to_string()))
        .unwrap_or_default();
    Some(Track { title, artist })
}
