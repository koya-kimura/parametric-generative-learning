function generateParams() {
    return {
        faceAspect: random(0.9, 1.3),
        tilt: random(-PI / 18, PI / 18),

        eyeSpacing: random(0.25, 0.45),
        eyeSize: random(0.06, 0.15),
        eyeOpen: random(0.5, 1.2),

        noseLength: random(0.08, 0.22),

        mouthWidth: random(0.3, 0.7),
        mouthHeight: random(0.03, 0.09),
        mouthCurve: random(-0.4, 0.4),

        hueSkin: random(15, 50),
        satSkin: random(10, 50),
        briSkin: random(50, 95),

        showCheek: random() < 0.7,
        cheekSatBoost: random(10, 40),
        cheekBriDrop: random(10, 30),

        eyeHue: random(0, 40),
        hairHue: random(0, 360),
        hairSat: random(20, 80),
        hairBri: random(20, 90),
        hairStyle: floor(random(0, 3))
    };
  }

function drawScene(g, p) {
    const {
        faceAspect, tilt,
        eyeSpacing, eyeSize, eyeOpen,
        noseLength,
        mouthWidth, mouthHeight, mouthCurve,
        hueSkin, satSkin, briSkin,
        showCheek, cheekSatBoost, cheekBriDrop,
        eyeHue, hairHue, hairSat, hairBri, hairStyle
    } = p;

    g.colorMode(HSB, 360, 100, 100, 100);
    g.background(0);
    g.push();
    g.translate(g.width / 2, g.height / 2);
    g.rotate(tilt);

    const faceW = g.width * 0.6;
    const faceH = faceW * faceAspect;

    // 髪（上半分〜おでこを覆う形）
    g.noStroke();
    g.fill(hairHue, hairSat, hairBri, 100);
    g.arc(0, -faceH * 0.5, faceW * 1.2, faceH * 1.0, PI, 0, g.CHORD);

    // 顔
    g.fill(hueSkin, satSkin, briSkin, 100);
    g.ellipse(0, 0, faceW, faceH);

    // 目
    const eyeY = -faceH * 0.2;
    const eyeXOffset = faceW * eyeSpacing * 0.5;
    const eyeW = g.width * eyeSize;
    const eyeH = eyeW * eyeOpen;

    // 白目
    g.fill(0, 0, 100);
    g.ellipse(-eyeXOffset, eyeY, eyeW, eyeH);
    g.ellipse(+eyeXOffset, eyeY, eyeW, eyeH);

    // 黒目（大きめ）+ ハイライト
    const pupilD = eyeW * 0.6;
    g.fill(eyeHue, 40, 20);
    g.ellipse(-eyeXOffset, eyeY, pupilD, pupilD);
    g.ellipse(+eyeXOffset, eyeY, pupilD, pupilD);

    // ハイライト（白い小円）
    const hlD = pupilD * 0.25;
    const hlOffset = pupilD * 0.2;
    g.fill(0, 0, 100, 80);
    g.ellipse(-eyeXOffset - hlOffset, eyeY - hlOffset, hlD);
    g.ellipse(+eyeXOffset - hlOffset, eyeY - hlOffset, hlD);

    // 鼻（ライン）
    g.stroke(0, 0, 30, 60);
    g.strokeWeight(g.width * 0.01);
    g.line(0, eyeY + eyeH / 2, 0, eyeY + eyeH / 2 + faceH * noseLength);
    g.noStroke();

    // 口（塗り）
    const mouthY = faceH * 0.28;
    const mouthW = faceW * mouthWidth;
    const mouthH = faceH * mouthHeight;
    g.fill((hueSkin + 10) % 360, satSkin + 20, briSkin - 20, 100);
    g.beginShape();
    for (let t = 0; t <= 1; t += 0.1) {
        const x = lerp(-mouthW / 2, mouthW / 2, t);
        const y = sin(t * PI) * mouthH * mouthCurve;
        g.vertex(x, mouthY + y);
    }
    g.endShape(g.CLOSE);

    // ほっぺ
    if (showCheek) {
        const cheekY = faceH * 0.12;
        const cheekX = faceW * 0.27;
        g.fill(hueSkin, satSkin + cheekSatBoost, briSkin - cheekBriDrop, 40);
        g.ellipse(-cheekX, cheekY, faceW * 0.18);
        g.ellipse(+cheekX, cheekY, faceW * 0.18);
    }

    g.pop();
  }