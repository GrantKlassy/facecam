use std::sync::{Arc, Mutex};

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

/// Now-playing is sourced from MPRIS (a Linux/D-Bus standard) which has no macOS
/// equivalent without private Apple frameworks, so this is a no-op stub: the track
/// stays `None` and the overlay shows "(no track)". The auto palette-on-track-change
/// feature simply never fires; palettes can still be cycled manually.
pub fn start() -> SharedTrack {
    Arc::new(Mutex::new(None))
}
