let canvasSize = 256;
let zip = new JSZip();
let csvData = [];

function setup() {
    noCanvas(); // 表示用キャンバス不要
}

// ランダムパラメータ生成
function generateParams() {
    return {
        radiusScale: random(0.2, 0.7),
        num: floor(random(5, 20)),
        hue: random(360),
    };
}

// CSVを生成して保存（ローカルに）
function generateAndSaveCSV() {
    const countInput = document.getElementById("csvCount");
    const count = parseInt(countInput.value) || 50;

    csvData = [];
    let header = "id,str,radiusScale,num,hue,score\n";
    csvData.push(header);

    for (let i = 0; i < count; i++) {
        let params = generateParams();
        let score = scorePreference(params);
        let line = `${i},${params.str},${params.radiusScale.toFixed(3)},${params.num},${params.hue.toFixed(1)},${score}\n`;
        csvData.push(line);
    }

    const blob = new Blob(csvData, { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "params.csv";
    a.click();
  }

// ローカルCSVを読み込んで画像ZIPを生成
function handleCSVUpload() {
    const fileInput = document.getElementById('csvFile');
    if (!fileInput.files.length) {
        alert("CSVファイルを選択してください！");
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = async function (event) {
        const text = event.target.result;
        const lines = text.trim().split("\n");
        const header = lines[0].split(",");
        const dataLines = lines.slice(1);

        for (let line of dataLines) {
            const values = line.split(",");
            const row = Object.fromEntries(header.map((h, i) => [h.trim(), values[i].trim()]));

            const params = {
                str: row.str,
                radiusScale: parseFloat(row.radiusScale),
                num: parseInt(row.num),
                hue: parseFloat(row.hue),
            };

            const g = createGraphics(canvasSize, canvasSize);
            drawScene(g, params);
            await new Promise(resolve => setTimeout(resolve, 10));
            const dataUrl = g.canvas.toDataURL();
            const base64 = dataUrl.split(",")[1];
            zip.file(`img_${row.id}.png`, base64, { base64: true });
        }

        zip.generateAsync({ type: "blob" }).then(blob => {
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "images.zip";
            a.click();
            zip = new JSZip(); // リセット
        });
    };

    reader.readAsText(file);
}