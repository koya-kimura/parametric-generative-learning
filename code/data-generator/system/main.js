
let zip = new JSZip();
let paramsList = [];
let canvasBuffers = [];
let loadedRows = [];
let currentCount = 0;
let canvasW = 300, canvasH = 300;
let spacing = 20;
let imageLimit = 100;
let canv;

function setup() {
  canv = createCanvas(canvasW * 3 + spacing * 2, canvasH);
  canv.parent("canvasWrapper");
  noLoop();
}

function draw() {
  background(17);
  for (let i = 0; i < 3; i++) {
    if (canvasBuffers[i]) {
      image(canvasBuffers[i], i * (canvasW + spacing), 0);
    }
  }
}

window.onload = () => {
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const csvInput = document.getElementById("csvInput");

  modeRadios.forEach(radio => {
    radio.addEventListener("change", () => {
      if (document.querySelector('input[name="mode"]:checked').value === "reannotate") {
        csvInput.style.display = "inline-block";
      } else {
        csvInput.style.display = "none";
      }
    });
  });
};

function startSession() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const saveImage = document.getElementById("saveImageCheckbox").checked;

  currentCount = 0;
  paramsList = [];
  zip = new JSZip();

  if (mode === "reannotate") {
    const fileInput = document.getElementById("csvInput");
    if (!fileInput.files[0]) {
      alert("CSVファイルを選択してください");
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      const text = e.target.result;
      const lines = text.trim().split("\n").slice(1);
      loadedRows = lines.map(line => {
        const [id, asp, hue, dipCount, circleCount, startAngle, rating, comment, timestamp] = line.split(",");
        return {
          id,
          asp: parseFloat(asp),
          hue: parseFloat(hue),
          dipCount: parseInt(dipCount),
          circleCount: parseInt(circleCount),
          startAngle: parseFloat(startAngle),
          rating: parseInt(rating),
          comment: comment.replace(/^"|"$/g, ""),
          timestamp: timestamp || new Date().toISOString()
        };
      });
      imageLimit = loadedRows.length;
      document.getElementById("progressText").innerText = `0 / ${imageLimit}`;
      for (let i = 0; i < 3; i++) loadAndDraw(i);
      redraw();
    };
    reader.readAsText(fileInput.files[0]);
  } else {
    imageLimit = parseInt(document.getElementById("imageLimitInput").value);
    document.getElementById("progressText").innerText = `0 / ${imageLimit}`;
    for (let i = 0; i < 3; i++) {
      canvasBuffers[i] = createGraphics(canvasW, canvasH);
      generateAndDraw(i);
    }
    redraw();
  }
}

function generateParams() {
  const dipCount = floor(random(2, 5) + map(pow(random(), 3), 0, 1, 1, 10));
  const circleCount = floor(map(random(), 0, 1, 1500, 120 * dipCount));
  return {
    asp: random(0.3, 0.5),
    hue: random(360),
    dipCount,
    circleCount,
    startAngle: random(TAU)
  };
}

function generateAndDraw(index) {
  const params = generateParams();
  const g = canvasBuffers[index];
  g.clear();
  drawScene(g, params);
  g.__params = params;
}

function loadAndDraw(index) {
  if (currentCount + index >= loadedRows.length) return;
  const row = loadedRows[currentCount + index];
  const g = createGraphics(canvasW, canvasH);
  drawScene(g, row);
  g.__params = row;
  canvasBuffers[index] = g;
}

function rateImage(index, score) {
  const comment = document.getElementById("comment" + index).value.replace(/\n/g, " ");
  const g = canvasBuffers[index];
  const params = g.__params;

  const id = params.id || (new Date().toISOString().slice(0,10).replace(/-/g, "") + "_" + String(paramsList.length + 1).padStart(4, "0"));
  const timestamp = new Date().toISOString();

  const saveImage = document.getElementById("saveImageCheckbox").checked;

  if (saveImage) {
    g.canvas.toBlob(blob => {
      zip.file(`image_${id}.png`, blob);
    }, 'image/png');
  }

  paramsList.push({
    id: id,
    ...params,
    rating: score,
    comment: comment,
    timestamp: timestamp
  });

  document.getElementById("comment" + index).value = "";

  if (loadedRows.length > 0 && currentCount + 3 < loadedRows.length) {
    const nextRow = loadedRows[currentCount + 3];
    const g2 = createGraphics(canvasW, canvasH);
    drawScene(g2, nextRow);
    g2.__params = nextRow;
    canvasBuffers[index] = g2;
  } else if (loadedRows.length === 0) {
    generateAndDraw(index);
  } else {
    canvasBuffers[index] = null;
  }

  currentCount++;
  document.getElementById("progressText").innerText = `${currentCount} / ${imageLimit}`;
  redraw();

  if (currentCount >= imageLimit) {
    forceDownload();
  }
}

function forceDownload() {
  let csv = "id,asp,hue,dipCount,circleCount,startAngle,rating,comment,timestamp\n";
  paramsList.forEach(p => {
    csv += `${p.id},${p.asp.toFixed(3)},${p.hue.toFixed(1)},${p.dipCount},${p.circleCount},${p.startAngle.toFixed(3)},${p.rating},"${p.comment}","${p.timestamp}"\n`;
  });

  const saveImage = document.getElementById("saveImageCheckbox").checked;

  if (!saveImage) {
    const blob = new Blob([csv], { type: "text/csv" });
    saveAs(blob, "metadata.csv");
    return;
  }

  zip.file("metadata.csv", csv);
  zip.generateAsync({ type: "blob" }).then(blob => {
    saveAs(blob, "images_and_metadata.zip");
  });
}
