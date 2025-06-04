// ranking.js

let graphicsLeft, graphicsRight, canvas;
let paramsList = [];
let currentPair = [];
let ranker;
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
  ranker = new RankingSystem(paramsList);

  nextComparison();
}

function nextComparison() {
  if (!ranker.hasNext()) {
    document.getElementById("progressText").innerText = "完了！";
    document.getElementById("progressBar").style.width = "100%";
    return;
  }

  currentPair = ranker.nextPair();
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
  ranker.vote(winnerSide);
  nextComparison();
}

function updateProgress() {
  const prog = ranker.getProgress();
  document.getElementById("progressText").innerText =
    `進捗: ${prog.done} / ${prog.total}`;
  document.getElementById("progressBar").style.width = `${prog.percentage}%`;
}

function downloadCSV() {
  const results = ranker.getResults();
  const ranks = ranker.getRankedIndices();

  const headers = ["id", ...Object.keys(paramsList[0]), "score", "rank"];
  const rows = paramsList.map((params, idx) => {
    const values = [`id_${idx.toString().padStart(4, "0")}`];
    for (let key of Object.keys(params)) {
      values.push(typeof params[key] === "number" ? params[key].toFixed(3) : params[key]);
    }
    values.push(results[idx]);
    values.push(ranks[idx]);
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