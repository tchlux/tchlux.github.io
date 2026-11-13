if (window.parent !== window) {
  document.body.classList.add('embedded');
  document.getElementById('open-page').hidden = false;
  document.querySelector('footer a').addEventListener('click', event => {
    event.preventDefault();
    window.parent.postMessage('close-flimsy-words', location.origin);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !document.getElementById('cold-bud').open) window.parent.postMessage('close-flimsy-words', location.origin);
  });
}

const sections = [...document.querySelectorAll('main section')];
const links = [...document.querySelectorAll('.contents a')];
const contents = document.querySelector('.contents');
let currentSection = '';

if (window.parent !== window) {
  for (const link of links) {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      history.replaceState(null, '', link.hash);
      document.getElementById(link.hash.slice(1)).scrollIntoView();
    });
  }
}

const updateSection = () => {
  const threshold = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--section-top')) + 60;
  let active = sections[0];
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= threshold) active = section;
  }
  if (active.id === currentSection) return;
  currentSection = active.id;
  document.getElementById('open-page').href = './#' + active.id;
  for (const link of links) {
    if (link.hash === '#' + active.id) {
      link.setAttribute('aria-current', 'location');
      contents.scrollLeft = link.offsetLeft - contents.clientWidth / 2 + link.offsetWidth / 2;
    } else {
      link.removeAttribute('aria-current');
    }
  }
};

window.addEventListener('scroll', updateSection, { passive: true });
window.addEventListener('resize', () => { currentSection = ''; updateSection(); });
updateSection();

const soundPlay = document.getElementById('sound-play');
const soundVolume = document.getElementById('sound-volume');
const soundStatus = document.getElementById('sound-status');
const soundStory = document.getElementById('cold-bud');
const soundStoryButton = document.getElementById('sound-story');
const soundPanel = document.querySelector('.sound-story');
let coldBudContext, coldBudGain, coldBudBuffer, coldBudSource;
let soundWanted = false;
let soundChosen = false;
let scrollAttempted = false;

soundPlay.hidden = false;
soundVolume.parentElement.hidden = false;

const updateSound = () => {
  const playing = soundWanted && coldBudContext?.state === 'running';
  const action = playing ? (coldBudSource ? 'Pause' : 'Cancel loading') : 'Play';
  soundPlay.textContent = playing ? (coldBudSource ? 'Ⅱ' : '…') : '▶';
  soundPlay.setAttribute('aria-label', action + ' Cold Bud soundtrack');
  soundPlay.title = action + ' Cold Bud soundtrack';
};

const pauseSound = () => {
  soundWanted = false;
  coldBudContext?.suspend();
  updateSound();
};

const playSound = async () => {
  soundWanted = true;
  try {
    if (!coldBudContext) {
      // Decode at 24 kHz to keep this ten-minute stereo loop modest in memory.
      coldBudContext = new AudioContext({ sampleRate: 24000 });
      coldBudGain = coldBudContext.createGain();
      coldBudGain.gain.value = Number(soundVolume.value);
      coldBudGain.connect(coldBudContext.destination);
      coldBudContext.addEventListener('statechange', updateSound);
    }
    // A scroll is not a sound permission on every browser. Keep Play available.
    await Promise.race([coldBudContext.resume(), new Promise(resolve => setTimeout(resolve, 300))]);
    if (!soundWanted) return;
    if (coldBudContext.state !== 'running') {
      soundWanted = false;
      soundStatus.textContent = 'Tap Play to hear the Cold Bud soundtrack.';
      updateSound();
      return;
    }
    updateSound();
    if (!coldBudSource) {
      soundStatus.textContent = 'Loading the Cold Bud soundtrack.';
      coldBudBuffer ||= fetch('../cold-bud/morning-soundtrack.m4a')
        .then(response => {
          if (!response.ok) throw new Error('Soundtrack download failed');
          return response.arrayBuffer();
        })
        .then(data => coldBudContext.decodeAudioData(data));
      const buffer = await coldBudBuffer;
      if (!soundWanted) return;
      if (!coldBudSource) {
        coldBudSource = coldBudContext.createBufferSource();
        coldBudSource.buffer = buffer;
        // The prepared recording loops directly at its sample boundary, without a fade or restart gap.
        coldBudSource.loop = true;
        coldBudSource.connect(coldBudGain);
        coldBudSource.start();
      }
    }
    updateSound();
    soundStatus.textContent = 'Cold Bud soundtrack playing on repeat.';
  } catch {
    coldBudBuffer = null;
    pauseSound();
    soundStatus.textContent = 'Unable to play. You can download the soundtrack from the Cold Bud story.';
  }
};

soundPlay.addEventListener('click', () => {
  soundChosen = true;
  if (soundWanted && coldBudContext?.state === 'running') pauseSound();
  else playSound();
});
soundVolume.addEventListener('input', () => {
  if (coldBudGain) coldBudGain.gain.value = Number(soundVolume.value);
});
window.addEventListener('scroll', () => {
  if (scrollY > 0 && !scrollAttempted) {
    scrollAttempted = true;
    if (!soundChosen) playSound();
  }
}, { passive: true });
window.addEventListener('pagehide', pauseSound);

let storyPinned = false;
let storyTimer;
const showStory = () => {
  clearTimeout(storyTimer);
  soundStory.open = true;
};
const hideStorySoon = () => {
  clearTimeout(storyTimer);
  if (!storyPinned) storyTimer = setTimeout(() => {
    if (!soundPanel.matches(':hover') && !soundPanel.matches(':focus-within')) soundStory.open = false;
  }, 220);
};
soundStoryButton.addEventListener('pointerenter', event => {
  if (event.pointerType === 'mouse') showStory();
});
soundStoryButton.addEventListener('pointerleave', hideStorySoon);
soundPanel.addEventListener('pointerenter', () => clearTimeout(storyTimer));
soundPanel.addEventListener('pointerleave', hideStorySoon);
soundStoryButton.addEventListener('click', event => {
  event.preventDefault();
  if (storyPinned && soundStory.open) soundStory.open = false;
  else { storyPinned = true; showStory(); }
});
document.getElementById('sound-story-close').addEventListener('click', () => { soundStory.open = false; soundStoryButton.focus(); });
document.addEventListener('pointerdown', event => {
  if (soundStory.open && !document.querySelector('.soundtrack').contains(event.target)) soundStory.open = false;
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && soundStory.open) {
    event.preventDefault();
    soundStory.open = false;
    soundStoryButton.focus();
  }
});
soundStory.addEventListener('toggle', () => {
  const open = soundStory.open;
  document.getElementById('open-page').href = open ? './#cold-bud' : './#' + currentSection;
  if (!open) storyPinned = false;
});
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== window.parent) return;
  if (event.data === 'pause-cold-bud') {
    soundChosen = true;
    pauseSound();
    soundStory.open = false;
  }
  if (event.data === 'show-cold-bud') { storyPinned = true; showStory(); }
});
if (location.hash === '#cold-bud') { storyPinned = true; showStory(); }

const scenes = document.querySelector('.sound-scenes');
const sceneTabs = [...document.querySelectorAll('.scene-tabs button')];
sceneTabs.forEach((button, index) => button.addEventListener('click', () => {
  scenes.scrollLeft = index * scenes.clientWidth;
}));
scenes.addEventListener('scroll', () => {
  const active = Math.round(scenes.scrollLeft / scenes.clientWidth);
  sceneTabs.forEach((button, index) => button.setAttribute('aria-pressed', String(index === active)));
}, { passive: true });
