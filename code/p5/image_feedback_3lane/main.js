let zip = new JSZip();
let paramsList = [];
let canvasBuffers = [];
let imageLimit = 100;
let currentCount = 0;
let canvasW = 300, canvasH = 300;
let spacing = 20;

function setup() {
  const canvas = createCanvas(canvasW * 3 + spacing * 2, canvasH);
  canvas.parent("canvasWrapper");
  for (let i = 0; i < 3; i++) {
    canvasBuffers[i] = createGraphics(canvasW, canvasH);
    generateAndDraw(i);
  }
  noLoop();
  redraw();
}

function draw() {
  background(17);
  for (let i = 0; i < 3; i++) {
    image(canvasBuffers[i], i * (canvasW + spacing), 0);
  }
}

function startSession() {
  imageLimit = parseInt(document.getElementById("imageLimitInput").value);
  currentCount = 0;
  paramsList = [];
  zip = new JSZip();
  for (let i = 0; i < 3; i++) {
    generateAndDraw(i);
  }
  document.getElementById("progressText").innerText = `0 / ${imageLimit}`;
  redraw();
}

function generateAndDraw(index) {
  const params = generateParams();
  const g = canvasBuffers[index];
  g.clear();
  drawScene(g, params);
  g.__params = params;
}

function rateImage(index, score) {
  const comment = document.getElementById("comment" + index).value.replace(/\n/g, " ");
  const g = canvasBuffers[index];
  const params = g.__params;

  const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  const id = dateStr + "_" + String(paramsList.length + 1).padStart(4, "0");

  g.canvas.toBlob(blob => {
    zip.file(`image_${id}.png`, blob);
  }, 'image/png');

  paramsList.push({
    id: id,
    ...params,
    rating: score,
    comment: comment
  });

  document.getElementById("comment" + index).value = "";
  generateAndDraw(index);
  currentCount++;
  document.getElementById("progressText").innerText = `${currentCount} / ${imageLimit}`;
  redraw();

  if (currentCount >= imageLimit) {
    forceDownload();
  }
}

function forceDownload() {
  let csv = "id,asp,hue,dipCount,circleCount,startAngle,rating,comment\n";
  paramsList.forEach(p => {
    csv += `${p.id},${p.asp.toFixed(3)},${p.hue.toFixed(1)},${p.dipCount},${p.circleCount},${p.startAngle.toFixed(3)},${p.rating},"${p.comment}"\n`;
  });
  zip.file("metadata.csv", csv);
  zip.generateAsync({ type: "blob" }).then(blob => {
    saveAs(blob, "3lane_images_and_metadata.zip");
  });
}
