const readingDialog = document.getElementById('reading-dialog');
const readingFrame = document.getElementById('reading-frame');
const writtenTalkLink = document.getElementById('written-talk-link');
const readingHash = '#read-flimsy-words';
let showColdBud = false;

const openReading = () => {
  if (readingDialog.open) return;
  if (!readingFrame.getAttribute('src')) readingFrame.src = writtenTalkLink.href + (showColdBud ? '#cold-bud' : '');
  readingDialog.showModal();
  document.body.classList.add('reading-open');
  document.getElementById('flimsy-words-video').pause();
  readingFrame.focus();
  if (showColdBud) readingFrame.contentWindow.postMessage('show-cold-bud', location.origin);
};

for (const link of [writtenTalkLink, document.getElementById('cold-bud-link')]) {
  link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    showColdBud = link.id === 'cold-bud-link';
    history.pushState({ reading: true }, '', readingHash);
    openReading();
  });
}

readingDialog.addEventListener('close', () => {
  if (readingDialog.open) return;
  readingFrame.contentWindow.postMessage('pause-cold-bud', location.origin);
  document.body.classList.remove('reading-open');
  if (location.hash === readingHash) {
    if (history.state?.reading) history.back();
    else history.replaceState(null, '', '#flimsy-words');
  }
});

window.addEventListener('popstate', () => {
  if (location.hash === readingHash) openReading();
  else if (readingDialog.open) readingDialog.close();
});

window.addEventListener('message', event => {
  if (event.origin === location.origin && event.source === readingFrame.contentWindow && event.data === 'close-flimsy-words' && readingDialog.open) readingDialog.close();
});

if (location.hash === readingHash) openReading();
