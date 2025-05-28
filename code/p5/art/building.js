function generateParams() {
    const base = min(windowWidth, windowHeight);

    const bw = random(0.2, 0.35) * base;
    const bh = random(0.25, 0.4) * base;
    const depth = random(0.05, 0.1) * base;
    const roofHeight = random(0.05, 0.1) * base;

    const numWindowsX = floor(random(2, 4));
    const numWindowsY = floor(random(2, 3));
    const doorWidth = random(0.05, 0.08) * base;
    const doorHeight = random(0.1, 0.15) * base;

    const baseHue = floor(random(0, 360 / 60)) * 60; // 原色系
    const mainColor = [baseHue, 100, 90];
    const sideColor = [(baseHue + 30) % 360, 100, 70];
    const roofColor = [(baseHue + 180) % 360, 100, 80];
    const windowColor = [0, 0, 100];
    const doorColor = [(baseHue + 240) % 360, 100, 60];

    return {
        bw, bh, depth, roofHeight,
        mainColor, sideColor, roofColor,
        windowColor, doorColor,
        numWindowsX, numWindowsY,
        doorWidth, doorHeight
    };
  }

function drawScene(g, params) {
    const {
        bw, bh, depth, roofHeight,
        mainColor, sideColor, roofColor,
        windowColor, doorColor,
        numWindowsX, numWindowsY,
        doorWidth, doorHeight
    } = params;

    g.push();
    g.translate(g.width * 0.45, g.height * 0.9);
    g.rectMode(CENTER);
    g.colorMode(HSB, 360, 100, 100, 100);
    g.noStroke();

    // --- 正面の壁 ---
    g.fill(...mainColor);
    g.rect(0, -bh / 2, bw, bh);

    // --- 側面（奥行き） ---
    g.fill(...sideColor);
    g.quad(
        bw / 2, -bh,
        bw / 2 + depth, -bh - depth * 0.5,
        bw / 2 + depth, 0 - depth * 0.5,
        bw / 2, 0
    );

    // --- 屋根（正面） ---
    g.fill(...roofColor);
    g.triangle(
        -bw / 2, -bh,
        bw / 2, -bh,
        0, -bh - roofHeight
    );

    // --- 屋根（奥行き部分） ---
    g.triangle(
        bw / 2, -bh,
        bw / 2 + depth, -bh - depth * 0.5,
        0, -bh - roofHeight
    );

    // --- ドア ---
    g.fill(...doorColor);
    const doorY = -doorHeight / 2;
    g.rect(0, doorY, doorWidth, doorHeight);

    // --- 窓（グリッド配置・ドアと重ならないように） ---
    g.fill(...windowColor);
    const winW = bw * 0.12;
    const winH = bh * 0.1;
    const xSpacing = bw / (numWindowsX + 1);
    const yTop = -bh + winH * 1.5;
    const yBottom = doorY - doorHeight * 0.8; // ドア上端の少し上まで

    const ySpacing = (yBottom - yTop) / (numWindowsY - 1);

    for (let i = 1; i <= numWindowsX; i++) {
        const x = -bw / 2 + i * xSpacing;
        for (let j = 0; j < numWindowsY; j++) {
            const y = yTop + j * ySpacing;
            g.rect(x, y, winW, winH);
        }
    }

    g.pop();
  }