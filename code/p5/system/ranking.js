let graphicsLeft, graphicsRight, canvas;
let paramsList = [];
let comparisons = [];
let currentPair = [];
let results = {};
let totalComparisons = 0;
let progressCount = 0;
let shouldDownloadImages = false;

function setup() {
  canvas = createCanvas(600, 300);
  canvas.parent("canvasWrapper");
  graphicsLeft = createGraphics(width / 2, height);
  graphicsRight = createGraphics(width / 2, height);

  noLoop();

  textAlign(CENTER, CENTER);

  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") vote("left");
    if (e.key === "ArrowRight") vote("right");
  });
}

function startRanking() {
  const n = parseInt(document.getElementById("numImages").value);
  shouldDownloadImages = document.getElementById("includeImages").checked;

  if (n < 2) {
    alert("最低でも2枚以上にしてください");
    return;
  }

  paramsList = Array.from({ length: n }, generateParams);
  results = {};
  progressCount = 0;

  comparisons = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      comparisons.push([i, j]);
    }
  }
  shuffle(comparisons, true);
  totalComparisons = comparisons.length;

  nextComparison();
}

function nextComparison() {
  if (comparisons.length === 0) {
    document.getElementById("progressText").innerText = "完了！";
    document.getElementById("progressBar").style.width = "100%";
    return;
  }

  currentPair = comparisons.pop();
  const [i, j] = currentPair;
  drawScene(graphicsLeft, paramsList[i]);
  drawScene(graphicsRight, paramsList[j]);
  redraw();
  updateProgress();
}

function draw() {
  background(0);
  image(graphicsLeft, 0, 0);
  image(graphicsRight, width / 2, 0);
}

function vote(winnerSide) {
  const [i, j] = currentPair;
  const winner = winnerSide === "left" ? i : j;

  if (!results[winner]) results[winner] = 0;
  results[winner] += 1;

  progressCount++;
  nextComparison();
}

function updateProgress() {
  document.getElementById("progressText").innerText =
    `進捗: ${progressCount} / ${totalComparisons}`;
  const percentage = (progressCount / totalComparisons) * 100;
  document.getElementById("progressBar").style.width = `${percentage}%`;
}

function downloadCSV() {
  const headers = ["id", ...Object.keys(paramsList[0]), "rating"];
  const rows = paramsList.map((params, idx) => {
    const rating = results[idx] || 0;
    const values = [`id_${idx.toString().padStart(4, "0")}`];
    for (let key of Object.keys(params)) {
      values.push(typeof params[key] === "number" ? params[key].toFixed(3) : params[key]);
    }
    values.push(rating);
    return values.join(",");
  });

  const csvContent = [headers.join(","), ...rows].join("\n");

  if (shouldDownloadImages) {
    const zip = new JSZip();
    zip.file("ranking_result.csv", csvContent);

    paramsList.forEach((params, idx) => {
      const g = createGraphics(256, 256);
      drawScene(g, params);
      const imgData = g.canvas.toDataURL("image/png");
      zip.file(`images/image_${idx.toString().padStart(4, "0")}.png`, imgData.split(",")[1], {
        base64: true
      });
    });

    zip.generateAsync({ type: "blob" }).then(function (content) {
      saveAs(content, "ranking_result.zip");
    });
  } else {
    const blob = new Blob([csvContent], { type: "text/csv" });
    saveAs(blob, "ranking_result.csv");
  }
}