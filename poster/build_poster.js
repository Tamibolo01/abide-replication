/* Build the A0 portrait poster (33.11 x 46.81 in) for the ABIDE project.
 *
 *   node build_poster.js <repo root> <output.pptx>
 *
 * Numbers come from the repo's results files (paths below); the pipeline A
 * accuracies are the README results table (results/ho_n871_summary.csv).
 * All text is Arial so a QuickLook/LibreOffice preview matches PowerPoint.
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(process.argv[2] || ".");
const OUT = process.argv[3] || "abide_poster_A0.pptx";

// ---------------------------------------------------------------- data
function readCsv(file) {
  const [head, ...rows] = fs.readFileSync(file, "utf8").trim().split("\n");
  const cols = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [cols[i], v])));
}
const history = readCsv(path.join(ROOT, "results/pitt10_check/loss_history.csv")).map((r) => ({
  step: +r.step, vv: +r.vv, vt: 0.5 * (+r.vt1 + +r.vt2), local: +r.local,
}));
const rounds = readCsv(path.join(ROOT, "results/pitt10_check/rounds.csv"));
const lastRound = rounds[rounds.length - 1];
const participants = readCsv(path.join(ROOT, "results/pitt10_check/participants.csv"));
const scores = participants.map((p) => ({ id: p.FILE_ID, label: +p.label, score: +lastRound[`score_${p.FILE_ID}`] }));
const sliceAcc = Math.round(100 * +lastRound.slice_accuracy);
const nSteps = +lastRound.steps_total;
const chance = Math.log(32);
const first10 = history.slice(0, 10), last10 = history.slice(-10);
const mean = (a, k) => a.reduce((s, r) => s + r[k], 0) / a.length;

// Pipeline A, leave-one-site-out, accuracy pooled over test subjects (README results table).
const pipelineA = [
  { label: "Tangent connectivity", value: 67.9 },
  { label: "Correlation", value: 66.6 },
  { label: "Partial correlation", value: 61.9 },
  { label: "Always guess \"control\"", value: 53.7 },
];

// ---------------------------------------------------------------- style
const INK = "1B2A41", INK2 = "52514E", MUTED = "898781", GRID = "E1E0D9";
const CARD = "EEF1F6", WHITE = "FFFFFF";
const BLUE = "2A78D6", ORANGE = "EB6834", AQUA = "1BAF7A";
const F = { title: 92, subtitle: 36, author: 30, h2: 44, body: 26, small: 22, stat: 64, statLabel: 22, chart: 22, num: 60 };

const pres = new pptxgen();
pres.defineLayout({ name: "A0_PORTRAIT", width: 33.11, height: 46.81 });
pres.layout = "A0_PORTRAIT";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
pres.title = "Can a computer tell autism from a resting brain scan?";
pres.author = "Tamilore Bolodeoku";
const slide = pres.addSlide();
slide.background = { color: WHITE };

const W = 33.11, M = 0.9, GAP = 0.6;
const COLW = (W - 2 * M - 2 * GAP) / 3; // 9.97
const X = [M, M + COLW + GAP, M + 2 * (COLW + GAP)];
const PAD = 0.45;

function text(t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontFace: "Arial", color: INK, valign: "top", align: "left", ...o });
}
function card(x, y, w, h, name) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.35, fill: { color: CARD }, line: { color: CARD, width: 0 }, objectName: name,
  });
}
function heading(t, x, y, w, lines = 1) {
  text(t, { x: x + PAD, y: y + PAD, w: w - 2 * PAD, h: 0.8 * lines, fontSize: F.h2, bold: true, lineSpacingMultiple: 1.05 });
}
function body(runs, x, y, w, h, size = F.body) {
  text(runs, { x: x + PAD, y, w: w - 2 * PAD, h, fontSize: size, color: INK, lineSpacingMultiple: 1.15 });
}
function bullets(items, x, y, w, h, size = F.body) {
  text(
    items.map((t, i) => ({ text: t, options: { bullet: { indent: 28 }, breakLine: i < items.length - 1, paraSpaceAfter: 10 } })),
    { x: x + PAD, y, w: w - 2 * PAD, h, fontSize: size, color: INK, lineSpacingMultiple: 1.12 },
  );
}
function stat(value, label, x, y, w, color = INK) {
  text(value, { x, y, w, h: 1.1, fontSize: F.stat, bold: true, color, align: "center", valign: "bottom" });
  text(label, { x, y: y + 1.15, w, h: 0.9, fontSize: F.statLabel, color: INK2, align: "center" });
}
function box(t, x, y, w, h, fill, color = WHITE, size = F.small) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.2, fill: { color: fill }, line: { color: fill, width: 0 } });
  text(t, { x: x + 0.12, y, w: w - 0.24, h, fontSize: size, color, align: "center", valign: "middle", bold: false });
}
function arrow(x, y) {
  slide.addShape(pres.ShapeType.rightArrow, { x, y, w: 0.45, h: 0.45, fill: { color: MUTED }, line: { color: MUTED, width: 0 } });
}
function numberBadge(n, x, y, size = 0.95) {
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: size, h: size, fill: { color: INK }, line: { color: INK, width: 0 } });
  text(String(n), { x, y, w: size, h: size, fontSize: 34, bold: true, color: WHITE, align: "center", valign: "middle" });
}

// ---------------------------------------------------------------- header
slide.addShape(pres.ShapeType.roundRect, {
  x: M, y: M, w: W - 2 * M, h: 5.9, rectRadius: 0.35, fill: { color: INK }, line: { color: INK, width: 0 }, objectName: "title block",
});
text("Can a computer tell autism\nfrom a resting brain scan?", {
  x: M + 0.7, y: M + 0.4, w: W - 2 * M - 1.4, h: 3.2, fontSize: F.title, bold: true, color: WHITE, valign: "middle",
});
text(
  "Replicating a brain-connectivity classifier and testing a picture-and-text model on ABIDE, 871 people scanned at 20 sites",
  { x: M + 0.7, y: M + 3.75, w: W - 2 * M - 1.4, h: 0.9, fontSize: F.subtitle, color: "CADCFC" },
);
text("Tamilore Bolodeoku  |  Yale University  |  github.com/Tamibolo01/abide-replication", {
  x: M + 0.7, y: M + 4.75, w: W - 2 * M - 1.4, h: 0.8, fontSize: F.author, color: WHITE,
});

const Y0 = M + 5.9 + GAP; // 7.4

// ---------------------------------------------------------------- column 1, card 1: the question
{
  const x = X[0], y = Y0, w = COLW, h = 10.2;
  card(x, y, w, h, "the question");
  heading("The question", x, y, w);
  body(
    "Resting-state fMRI records how the blood-oxygen signal rises and falls across the brain while a person lies still " +
      "for a few minutes. ABIDE is a public collection of such scans, about half from people with an autism diagnosis.\n\n" +
      "Can a computer learn, from the scans alone, who has the diagnosis? And does what it learns still work at a hospital " +
      "it has never seen?",
    x, y + 1.5, w, 5.6,
  );
  const sw = (w - 2 * PAD) / 3;
  stat("871", "people scanned", x + PAD, y + 7.5, sw, BLUE);
  stat("20", "sites (hospitals, universities)", x + PAD + sw, y + 7.5, sw, BLUE);
  stat("46%", "with an autism diagnosis", x + PAD + 2 * sw, y + 7.5, sw, BLUE);
}

// ---------------------------------------------------------------- column 1, card 2: two approaches
{
  const x = X[0], y = Y0 + 10.2 + GAP, w = COLW, h = 15.2;
  card(x, y, w, h, "two approaches");
  heading("Two ways to ask the computer", x, y, w);
  const bw = 1.95, aw = 0.45, bh = 1.9;
  const row = (label, color, boxes, ry, note) => {
    text(label, { x: x + PAD, y: ry, w: w - 2 * PAD, h: 0.7, fontSize: F.body, bold: true, color });
    let bx = x + PAD;
    boxes.forEach((t, i) => {
      box(t, bx, ry + 0.7, bw, bh, i === boxes.length - 1 ? color : WHITE, i === boxes.length - 1 ? WHITE : INK, 20);
      bx += bw;
      if (i < boxes.length - 1) { arrow(bx, ry + 0.7 + (bh - aw) / 2); bx += aw; }
    });
    body(note, x, ry + 0.7 + bh + 0.2, w, 2.9, F.small);
  };
  row("A. Connectivity fingerprint", BLUE,
    ["Scan: about 100 brain regions", "Do pairs of regions rise and fall together?", "5,050 numbers per person", "Simple classifier"],
    y + 1.55,
    "Replicates Abraham et al. (2017). A weighted sum of the 5,050 numbers decides. Three ways of measuring \"together\" are " +
      "compared; the best, tangent embedding, first lines every scan up against the group average.");
  row("B. Pictures and words", AQUA,
    ["One slice at one instant", "A short report from the record", "Model learns to match picture to report", "Which report fits best?"],
    y + 7.4,
    "Adapts MaMA (Du et al., 2024), a mammography model. No time series at all: the model only ever sees single pictures. " +
      "A picture is a map of where the signal is above or below its usual level at that instant, not anatomy. Personal details " +
      "in the reports are hidden 80% of the time so the model cannot cheat by memorising them.");
  // what the model actually sees: three real input slices of one participant (data/slices/Pitt_0050003.npz)
  ["axial", "coronal", "sagittal"].forEach((plane, i) => {
    slide.addImage({ path: path.join(__dirname, `slice_${plane}.png`), x: x + PAD + i * 2.0, y: y + 13.25, w: 1.8, h: 1.8, objectName: `${plane} slice`, altText: `${plane} slice of one fMRI volume at one time point, 76 by 76 pixels` });
  });
  text("What the model actually sees: one instant of one scan, cut three ways (76 x 76 pixels each; mid-grey = signal at its usual level).", {
    x: x + PAD + 6.0, y: y + 13.25, w: w - 2 * PAD - 6.0, h: 1.8, fontSize: 19, color: INK2, valign: "middle",
  });
}

// ---------------------------------------------------------------- column 2, card 3: result 1
{
  const x = X[1], y = Y0, w = COLW, h = 15.4;
  card(x, y, w, h, "result 1");
  heading("Result 1: the fingerprint works at new hospitals", x, y, w, 2);
  body("Trained on 19 sites, tested on the 20th, repeated for every site. Share of people classified correctly, pooled over all 871:",
    x, y + 2.3, w, 1.3, F.small);
  slide.addChart(pres.ChartType.bar, [{ name: "Accuracy", labels: pipelineA.map((d) => d.label), values: pipelineA.map((d) => d.value) }], {
    x: x + PAD, y: y + 3.7, w: w - 2 * PAD, h: 6.9,
    barDir: "col", barGapWidthPct: 60,
    chartColors: [BLUE, BLUE, BLUE, MUTED],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"', dataLabelFontSize: F.chart, dataLabelColor: INK, dataLabelFontFace: "Arial",
    catAxisLabelFontSize: F.chart, catAxisLabelColor: INK, catAxisLabelFontFace: "Arial", catAxisLineShow: false,
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, valAxisLabelFontSize: F.chart, valAxisLabelColor: INK2, valAxisLabelFontFace: "Arial",
    valAxisLabelFormatCode: '0"%"', valAxisLineShow: false,
    valGridLine: { color: GRID, size: 1 }, catGridLine: { style: "none" },
    showLegend: false, showTitle: false, plotArea: { fill: { color: CARD } }, chartArea: { fill: { color: CARD } },
    objectName: "pipeline A accuracy",
  });
  bullets(
    [
      "About two in three people correct at a hospital the model never saw, against 54% from always guessing \"control\". This matches the published paper, down to tangent embedding being the best measure.",
      "Within the same sites (10-fold cross-validation) the best setting reaches 68%.",
    ],
    x, y + 10.9, w, 4.2, F.small,
  );
}

// ---------------------------------------------------------------- column 2, card 4: the audit
{
  const x = X[1], y = Y0 + 15.4 + GAP, w = COLW, h = 10.0;
  card(x, y, w, h, "audit");
  heading("What checking our own code taught us", x, y, w, 2);
  body("A first version found no advantage for tangent embedding, unlike the paper. Before blaming the paper we audited our pipeline, " +
    "one change at a time on the same people, and found three mistakes of our own:", x, y + 2.3, w, 1.9, F.small);
  const items = [
    ["Signals not standardised", "the library only did it for one of the three measures"],
    ["One setting fixed by hand", "instead of tuned on the training people"],
    ["Wrong \"chance\" level", "random guessing (51%) instead of always guessing the larger group (54%)"],
  ];
  items.forEach(([t, d], i) => {
    const iy = y + 4.4 + i * 1.3;
    numberBadge(i + 1, x + PAD, iy, 0.8);
    text([{ text: t + ": ", options: { bold: true } }, { text: d }], { x: x + PAD + 1.05, y: iy - 0.02, w: w - 2 * PAD - 1.05, h: 1.2, fontSize: F.small, color: INK });
  });
  body([{ text: "Fixing them changed the conclusion. ", options: { bold: true } }, { text: "Check your own pipeline before you argue with a paper." }],
    x, y + 8.5, w, 1.2, F.small);
}

// ---------------------------------------------------------------- column 3, card 5: result 2
{
  const x = X[2], y = Y0, w = COLW, h = 19.5;
  card(x, y, w, h, "result 2");
  heading("Result 2: the picture-and-text model works as machinery", x, y, w, 2);
  body("Before days of cluster time, a sanity check: give the model 10 people from one site (5 autism, 5 control) and ask it to " +
    "learn them by heart. A model that cannot even memorise ten people it has seen has a broken pipeline, not a hard problem.",
    x, y + 2.3, w, 2.4, F.small);
  const sw = (w - 2 * PAD) / 3;
  stat("10/10", "people learned", x + PAD, y + 4.9, sw, AQUA);
  stat(`${sliceAcc}%`, "of their single pictures", x + PAD + sw, y + 4.9, sw, AQUA);
  stat("0.00", "change after saving and reloading", x + PAD + 2 * sw, y + 4.9, sw, AQUA);

  text("Loss at every training step. Falling = learning; grey = random guessing.", {
    x: x + PAD, y: y + 7.2, w: w - 2 * PAD, h: 0.8, fontSize: 20, color: INK2,
  });
  const xs = history.map((r) => r.step);
  slide.addChart(pres.ChartType.scatter, [
    { name: "X-Axis", values: xs },
    { name: "picture vs report", values: history.map((r) => +r.vt.toFixed(3)) },
    { name: "patch vs sentence (on from step 50)", values: history.map((r) => +r.local.toFixed(3)) },
    { name: "picture vs its own second view", values: history.map((r) => +r.vv.toFixed(3)) },
    { name: "random guessing, ln 32", values: xs.map(() => +chance.toFixed(3)) },
  ], {
    x: x + PAD, y: y + 8.1, w: w - 2 * PAD, h: 5.6,
    lineSize: 2.5, lineDataSymbol: "none", chartColors: [BLUE, ORANGE, AQUA, MUTED], catAxisMinVal: 0, catAxisMaxVal: 250, catAxisMajorUnit: 50,
    showLegend: true, legendPos: "b", legendFontSize: 20, legendColor: INK, legendFontFace: "Arial",
    catAxisLabelFontSize: F.chart, catAxisLabelColor: INK2, catAxisLabelFontFace: "Arial", catAxisTitle: "training step", showCatAxisTitle: true, catAxisTitleFontSize: 20, catAxisTitleColor: INK2,
    valAxisLabelFontSize: F.chart, valAxisLabelColor: INK2, valAxisLabelFontFace: "Arial", valAxisMinVal: 0, valAxisMaxVal: 4, valAxisMajorUnit: 1, valAxisTitle: "loss", showValAxisTitle: true, valAxisTitleFontSize: 20, valAxisTitleColor: INK2,
    valGridLine: { color: GRID, size: 1 }, catGridLine: { style: "none" }, valAxisLineShow: false, catAxisLineShow: false,
    showTitle: false, plotArea: { fill: { color: CARD } }, chartArea: { fill: { color: CARD } },
    objectName: "training losses",
  });

  text("Final score per person after training. Positive = \"autism\" fits better.", {
    x: x + PAD, y: y + 13.9, w: w - 2 * PAD, h: 0.8, fontSize: 20, color: INK2,
  });
  slide.addChart(pres.ChartType.bar, [{ name: "score", labels: scores.map((s) => s.id.replace("Pitt_00", "")), values: scores.map((s) => +s.score.toFixed(2)) }], {
    x: x + PAD, y: y + 14.8, w: w - 2 * PAD, h: 3.7,
    barDir: "col", barGapWidthPct: 40, chartColors: [AQUA], invertedColors: [ORANGE],
    catAxisLabelFontSize: 18, catAxisLabelColor: INK2, catAxisLabelFontFace: "Arial", catAxisLabelPos: "low", catAxisLineShow: false,
    valAxisLabelFontSize: 18, valAxisLabelColor: INK2, valAxisLabelFontFace: "Arial", valAxisMinVal: -1.2, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.4, valAxisLabelFormatCode: "0.0",
    valGridLine: { color: GRID, size: 1 }, catGridLine: { style: "none" }, valAxisLineShow: false,
    showValue: false, showLegend: false, showTitle: false, plotArea: { fill: { color: CARD } }, chartArea: { fill: { color: CARD } },
    objectName: "per-participant scores",
  });
  text([{ text: "■ ", options: { color: AQUA } }, { text: "autism diagnosis     " }, { text: "■ ", options: { color: ORANGE } }, { text: "control" }], {
    x: x + PAD, y: y + 18.6, w: w - 2 * PAD, h: 0.5, fontSize: 20, color: INK2, align: "center",
  });
}

// ---------------------------------------------------------------- column 3, card 6: at real scale
{
  const x = X[2], y = Y0 + 19.5 + GAP, w = COLW, h = 5.9;
  card(x, y, w, h, "at scale");
  heading("At real scale: not yet", x, y, w);
  body("On 50 people from one site with a few minutes of training, the model is no better than guessing (52% chance). " +
    "That is expected at this scale and is not a verdict. The slowest part of training is matching pictures to reports, because once " +
    "personal details are hidden most reports read the same.", x, y + 1.5, w, 4.0, F.small);
}

// ---------------------------------------------------------------- next steps band
{
  const x = M, y = Y0 + 26.0 + GAP, w = W - 2 * M, h = 4.9;
  card(x, y, w, h, "next steps");
  heading("Next steps", x, y, w);
  const steps = [
    ["Run the model at scale", "all 50 PITT people with proper cross-validation, then all 20 sites, on the Yale cluster with longer training."],
    ["Compare the two fairly", "same people, same splits, including the leave-one-site-out test that matters clinically."],
    ["Richer text reports", "hide fewer details or add clinical fields, so pictures can be matched to reports."],
    ["Back to the biology", "which region pairs carry the most weight, and do they match autism networks in the literature?"],
    ["Write it up", "once step 1 has run."],
  ];
  const cw = (w - 2 * PAD - 4 * 0.4) / 5;
  steps.forEach(([t, d], i) => {
    const sx = x + PAD + i * (cw + 0.4);
    numberBadge(i + 1, sx, y + 1.55, 0.95);
    text(t, { x: sx + 1.15, y: y + 1.55, w: cw - 1.15, h: 0.95, fontSize: F.body, bold: true, valign: "middle" });
    text(d, { x: sx, y: y + 2.65, w: cw, h: 2.0, fontSize: F.small, color: INK2 });
  });
}

// ---------------------------------------------------------------- trust band
{
  const x = M, y = Y0 + 26.0 + GAP + 4.9 + GAP, w = W - 2 * M, h = 3.4;
  card(x, y, w, h, "trust");
  heading("How we make sure the numbers can be trusted", x, y, w);
  const items = [
    ["No peeking.", " Every number comes from people the model never saw during training, and anything learned from data (even an average) is learned from the training group only. Automated tests check this every time the code changes."],
    ["Quality gates.", " Tests and style checks must pass before a change is kept, and every reported number is tied to the exact command and code version that produced it."],
    ["Honest chance.", " \"Chance\" is what you get by always guessing the larger group (54%), not 50%."],
  ];
  const cw = (w - 2 * PAD - 2 * 0.5) / 3;
  items.forEach(([t, d], i) => {
    text([{ text: t, options: { bold: true } }, { text: d }], { x: x + PAD + i * (cw + 0.5), y: y + 1.45, w: cw, h: 1.8, fontSize: F.small, color: INK });
  });
}

// ---------------------------------------------------------------- footer: glossary + references
{
  const y = Y0 + 26.0 + GAP + 4.9 + GAP + 3.4 + 0.45;
  text(
    [
      { text: "Cross-validation: ", options: { bold: true } }, { text: "train on most people, test on the rest, repeated so everyone is tested once.   " },
      { text: "Leave-one-site-out: ", options: { bold: true } }, { text: "train on 19 sites, test on the 20th.   " },
      { text: "Leakage: ", options: { bold: true } }, { text: "any way test information can influence training; it inflates results.   " },
      { text: "Loss: ", options: { bold: true } }, { text: "the model's running error score; falling means learning." },
    ],
    { x: M, y, w: W - 2 * M, h: 1.1, fontSize: 19, color: INK2 },
  );
  text(
    "Abraham A. et al. (2017) Deriving reproducible biomarkers from multi-site resting-state data: the ABIDE autism dataset, NeuroImage.   " +
      "Du Y., Onofrey J., Dvornek N.C. (2024) MaMA: multi-view and multi-scale alignment for contrastive language-image pre-training in mammography, IPMI 2025.   " +
      "Data: ABIDE Preprocessed Connectomes Project (C-PAC pipeline, quality-checked sample).",
    { x: M, y: y + 1.2, w: W - 2 * M, h: 0.9, fontSize: 17, color: MUTED },
  );
}

pres.writeFile({ fileName: OUT }).then((f) => {
  console.log("wrote", f);
  console.log(`losses: picture-report ${mean(first10, "vt").toFixed(2)} -> ${mean(last10, "vt").toFixed(2)}, ` +
    `picture-picture ${mean(first10, "vv").toFixed(2)} -> ${mean(last10, "vv").toFixed(2)}, steps ${nSteps}, slice accuracy ${sliceAcc}%`);
});
