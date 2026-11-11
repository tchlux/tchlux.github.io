const recording = document.getElementById('flimsy-words-video');
const recordingPlay = document.getElementById('recording-play');
const recordingError = document.getElementById('recording-error');
const recordingSource = 'flimsy-words/index.m3u8';
const nativeHls = recording.canPlayType('application/vnd.apple.mpegurl');

recording.controls = false;
recordingPlay.hidden = false;
recordingPlay.addEventListener('click', () => {
  recordingPlay.hidden = true;
  recording.controls = true;
  if (nativeHls && 'ManagedMediaSource' in window) {
    recording.src = recordingSource;
  } else if (window.Hls && Hls.isSupported()) {
    const stream = new Hls({ maxBufferLength: 18 });
    stream.loadSource(recordingSource);
    stream.attachMedia(recording);
    stream.on(Hls.Events.ERROR, (event, data) => {
      if (data.fatal) recordingError.hidden = false;
    });
  } else if (nativeHls) {
    recording.src = recordingSource;
  }
  recording.play().catch(() => { recordingError.hidden = false; });
}, { once: true });

recording.addEventListener('error', () => { recordingError.hidden = false; });
