function generateParams() {
    return {
        radiusScale: random(0.2, 0.7),
        num: floor(random(5, 20)),
        hue: random(360),
    }
}

function drawScene(g, params) {
    g,push();
    g.textFont("Helvetica");
    g.colorMode(HSB, 360, 100, 100);
    g.translate(g.width / 2, g.height / 2);
    for(let i = 0; i < params.num; i++) {
        g.push();
        g.rotate(TWO_PI * (i / params.num));
        g.fill(params.hue, 100, 100);
        g.textSize(min(width, height) * 0.15);
        g.textAlign(CENTER, CENTER);
        g.translate(min(width, height) * 0.5 * params.radiusScale, 0);
        g.rotate(PI/2);
        g.text("A", 0, 0);
        g.pop();
    }
    g.pop();
}

const idealPoint = {
    hue: 0,
    num: 12.5,
    radiusScale: 0.7
};

const scales = {
    hue: 180,        // 差が最大でも180度（反対側）
    num: 7.5,
    radiusScale: 0.25
};

function circularDistance(a, b, max = 360) {
    const diff = Math.abs(a - b) % max;
    return Math.min(diff, max - diff);
}

function scorePreference(params) {
    const dh = circularDistance(params.hue, idealPoint.hue) / scales.hue;
    const dn = (params.num - idealPoint.num) / scales.num;
    const dr = (params.radiusScale - idealPoint.radiusScale) / scales.radiusScale;

    const distance = Math.sqrt(dh * dh + dn * dn + dr * dr);
    const sigma = 1.0;
    const score = Math.exp(- (distance * distance) / (2 * sigma * sigma));

    return score.toFixed(3);
}