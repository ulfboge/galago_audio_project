/* Galago demo map — loaded after Leaflet via galago_map_boot (see upload_predict_gradio.py). */
(function () {
  if (window.__galagoMapBooted) return;

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

  function initMap() {
    window.__galagoMapBooted = true;
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

    function fixSize() {
      try { map.invalidateSize(); } catch (e) {}
    }
    setTimeout(fixSize, 100);
    setTimeout(fixSize, 800);
    window.addEventListener("resize", fixSize);
  }

  function waitForMap(attempts) {
    if (document.getElementById("galago-map")) {
      initMap();
      return;
    }
    if (attempts > 0) {
      setTimeout(function () { waitForMap(attempts - 1); }, 250);
    }
  }

  waitForMap(80);
})();
