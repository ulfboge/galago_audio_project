/* Galago demo map — see upload_predict_gradio.py (Blocks head/js + this file). */
(function () {
  function setGradioField(elemId, value) {
    var root = document.getElementById(elemId);
    if (!root) return;
    var inp = root.querySelector("textarea, input");
    if (!inp) return;
    inp.value = value;
    inp.dispatchEvent(new Event("input", { bubbles: true }));
    inp.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function pushCoords(lat, lon) {
    var la = Number(lat).toFixed(6);
    var lo = Number(lon).toFixed(6);
    setGradioField("galago_paste_coords", la + ", " + lo);
    setGradioField("galago_lat", la);
    setGradioField("galago_lon", lo);
  }

  function fixMapSize(map, el) {
    try {
      if (el) {
        el.style.height = "400px";
        el.style.width = "100%";
      }
      map.invalidateSize({ pan: false });
    } catch (e) {}
  }

  function initMap() {
    if (window.__galagoMapBooted) return;
    var out = document.getElementById("galago-coord-out");
    if (typeof L === "undefined") {
      if (out) {
        out.textContent =
          "Kartan kunde inte laddas. Välj förvald plats eller fyll i lat/long manuellt.";
      }
      return;
    }
    var el = document.getElementById("galago-map");
    if (!el) return;

    window.__galagoMapBooted = true;
    el.style.height = "400px";
    el.style.width = "100%";

    var map = L.map(el, { scrollWheelZoom: true }).setView([-5, 25], 4);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png", {
      maxZoom: 20,
      subdomains: "abcd",
      attribution: "&copy; OpenStreetMap &copy; CARTO",
    }).addTo(map);

    var marker = null;
    var lastLa = null;
    var lastLo = null;
    var copyBtn = document.getElementById("galago-copy-coords");
    var copyMsg = document.getElementById("galago-copy-msg");

    map.on("click", function (e) {
      lastLa = e.latlng.lat.toFixed(6);
      lastLo = e.latlng.lng.toFixed(6);
      if (marker) map.removeLayer(marker);
      marker = L.marker(e.latlng).addTo(map);
      if (out) {
        out.innerHTML =
          "<strong>Latitude:</strong> " + lastLa + "<br><strong>Longitude:</strong> " + lastLo +
          "<br><small>Fälten nedan uppdaterades automatiskt.</small>";
      }
      if (copyBtn) copyBtn.disabled = false;
      if (copyMsg) copyMsg.textContent = "";
      pushCoords(lastLa, lastLo);
    });

    if (copyBtn) {
      copyBtn.onclick = function () {
        if (lastLa === null) return;
        var line = lastLa + "\t" + lastLo;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(line).then(function () {
            if (copyMsg) copyMsg.textContent = "Kopierat!";
          }).catch(function () {
            if (copyMsg) copyMsg.textContent = "Kunde inte kopiera — markera siffrorna manuellt.";
          });
        } else if (copyMsg) {
          copyMsg.textContent = "Ingen clipboard-API — kopiera från rutan ovan.";
        }
      };
    }

    if (out) {
      out.textContent = "Klicka på kartan för lat/lon (WGS84, decimalgrader).";
    }

    function reflow() {
      fixMapSize(map, el);
    }
    reflow();
    requestAnimationFrame(reflow);
    setTimeout(reflow, 100);
    setTimeout(reflow, 400);
    setTimeout(reflow, 1000);
    setTimeout(reflow, 2500);
    window.addEventListener("resize", reflow);
    if (typeof ResizeObserver !== "undefined") {
      var ro = new ResizeObserver(reflow);
      ro.observe(el);
      var wrap = document.getElementById("galago-map-wrap");
      if (wrap) ro.observe(wrap);
    }
  }

  function sized(el) {
    return el && el.clientWidth >= 80 && el.clientHeight >= 80;
  }

  function waitForSizedMap(attempts) {
    var el = document.getElementById("galago-map");
    if (sized(el)) {
      initMap();
      return;
    }
    if (el && attempts > 0) {
      el.style.height = "400px";
      el.style.width = "100%";
    }
    if (attempts > 0) {
      setTimeout(function () { waitForSizedMap(attempts - 1); }, 200);
      return;
    }
    if (el) {
      initMap();
      return;
    }
    var out = document.getElementById("galago-coord-out");
    if (out) out.textContent = "Kartcontainern hittades inte.";
  }

  waitForSizedMap(100);
})();
