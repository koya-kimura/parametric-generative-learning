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

function drawScene(g, params) {
  const { asp, hue, dipCount, circleCount, startAngle } = params;
  g.colorMode(HSB, 360, 100, 100, 100);
  g.background(0);
  g.push();
  g.translate(g.width / 2, g.height / 2);
  g.blendMode(ADD);
  const baseSize = g.width;
  const strokeW = baseSize * 0.0025;

  for (let t = 0; t < circleCount; t++) {
    const angle = startAngle + (TAU * t) / circleCount;
    const radius = baseSize * map(sin((TAU * t * dipCount) / circleCount), -1, 1, 0.25, 0.4);
    const x = radius * cos(angle);
    const y = radius * asp * sin(angle);
    const yNorm = map(y, 0, g.height, g.height, 0) / g.height;
    const size = baseSize * map(pow(yNorm, 2), 0, 1, 0.2, 0.18);
    const alpha = map(pow(yNorm, 3), 0, 1, 60, 30);
    g.noFill();
    g.strokeWeight(strokeW);
    g.stroke(hue, random(70, 90), random(70, 90), alpha);
    g.circle(x, y, size);
  }

  g.blendMode(BLEND);
  g.pop();
}
