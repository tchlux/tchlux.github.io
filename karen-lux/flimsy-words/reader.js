if (window.parent !== window) {
  document.body.classList.add('embedded');
  document.getElementById('open-page').hidden = false;
  document.querySelector('footer a').addEventListener('click', event => {
    event.preventDefault();
    window.parent.postMessage('close-flimsy-words', location.origin);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') window.parent.postMessage('close-flimsy-words', location.origin);
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
