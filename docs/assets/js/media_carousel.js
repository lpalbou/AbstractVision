(() => {
  function initCarousel(root) {
    if (!root) return;
    const track = root.querySelector(".af_media_carousel__track");
    const prev = root.querySelector(".af_media_carousel__btn--prev");
    const next = root.querySelector(".af_media_carousel__btn--next");
    if (!track || !prev || !next) return;

    function stepPx() {
      const first = track.querySelector(".af_media_carousel__item");
      if (!first) return Math.max(320, Math.floor(track.clientWidth * 0.85));
      const rect = first.getBoundingClientRect();
      // gap is 16px in CSS; keep it in sync.
      return Math.max(240, Math.floor(rect.width + 16));
    }

    function updateDisabled() {
      const maxScrollLeft = track.scrollWidth - track.clientWidth;
      prev.disabled = track.scrollLeft <= 2;
      next.disabled = track.scrollLeft >= maxScrollLeft - 2;
    }

    prev.addEventListener("click", () => {
      track.scrollBy({ left: -stepPx(), behavior: "smooth" });
    });
    next.addEventListener("click", () => {
      track.scrollBy({ left: stepPx(), behavior: "smooth" });
    });

    track.addEventListener("scroll", () => {
      window.requestAnimationFrame(updateDisabled);
    });

    // Initial state after layout.
    window.setTimeout(updateDisabled, 0);
  }

  function initAll() {
    document.querySelectorAll(".af_media_carousel").forEach((root) => initCarousel(root));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();

