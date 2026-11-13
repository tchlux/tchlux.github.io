const readingDialog = document.getElementById('reading-dialog');
const readingFrame = document.getElementById('reading-frame');
const writtenTalkLink = document.getElementById('written-talk-link');
const readingHash = '#read-flimsy-words';

const openReading = () => {
  if (readingDialog.open) return;
  if (!readingFrame.getAttribute('src')) readingFrame.src = writtenTalkLink.href;
  readingDialog.showModal();
  document.body.classList.add('reading-open');
  readingFrame.focus();
};

writtenTalkLink.addEventListener('click', event => {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  history.pushState({ reading: true }, '', readingHash);
  openReading();
});

readingDialog.addEventListener('close', () => {
  if (readingDialog.open) return;
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
