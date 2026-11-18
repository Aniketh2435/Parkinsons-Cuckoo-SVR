"""Fill the 13-slide BMSCE template skeleton, reusing the sample's own text styles (Times New Roman bullets)."""
import copy
import sys

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

SKELETON, OUT = sys.argv[1], sys.argv[2]
FIG = "/Users/anikethsnadig/BIO_AAT/outputs/"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
FONT = "Times New Roman"

prs = Presentation(SKELETON)
S = list(prs.slides)


def shape(slide, suffix):
    return next(sh for sh in slide.shapes if sh.name.endswith(suffix))


def _restyle(p, size):
    """Set every run of a cloned paragraph to Times New Roman at `size` pt."""
    for rpr in p.iter(f"{A}rPr", f"{A}endParaRPr"):
        rpr.set("sz", str(size * 100))
        for tag in ("latin", "ea", "cs", "sym"):
            el = rpr.find(f"{A}{tag}")
            if el is None:
                el = rpr.makeelement(f"{A}{tag}", {})
                rpr.append(el)
            el.set("typeface", FONT)
    bus = p.find(f"{A}pPr/{A}buSzPts")
    if bus is not None:
        bus.set("val", str(size * 100))


def _with_text(proto, text, size):
    p = copy.deepcopy(proto)
    runs = p.findall(f"{A}r")
    for r in runs[1:]:
        p.remove(r)
    runs[0].find(f"{A}t").text = text
    _restyle(p, size)
    return p


def fill(box, items, h=36, s=32):
    """items: ("h", text) top-level bullet, ("s", text) sub-bullet, ("b",) blank line.
    Prototypes are the box's own first level-0 and level-1 bullet paragraphs."""
    body = box.text_frame._txBody
    paras = body.findall(f"{A}p")
    top = next(p for p in paras if p.find(f"{A}pPr").get("lvl") == "0" and p.find(f"{A}pPr/{A}buChar") is not None)
    sub = next(p for p in paras if p.find(f"{A}pPr").get("lvl") == "1")
    blank = copy.deepcopy(top)
    ppr = blank.find(f"{A}pPr")
    ppr.remove(ppr.find(f"{A}buChar"))
    ppr.append(ppr.makeelement(f"{A}buNone", {}))
    for p in paras:
        body.remove(p)
    for it in items:
        if it[0] == "b":
            body.append(_with_text(blank, "", s // 2))
        else:
            body.append(_with_text(top if it[0] == "h" else sub, it[1], h if it[0] == "h" else s))


def set_title(slide, text):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.top < Inches(0.6) and sh.left > Inches(4):
            sh.left, sh.width = Inches(3.4), Inches(15.2)
            sh.text_frame.paragraphs[0].runs[0].text = text
            return
    raise ValueError("no title box")


def body(slide):
    return shape(slide, "97;p3")


# ================================================================== 1 Title slide
s = S[0]
t = shape(s, "70;p1")
ps = t.text_frame.paragraphs
ps[0].runs[0].text = "Bio Inspired Algorithms"
ps[1].runs[0].text = "23DS7BSBIO"
ps[1].runs[1].text = ""
ps[3].runs[0].text = ("Parkinson's Disease Severity Prediction from Voice Recordings using "
                      "Support Vector Regression Optimised with Cuckoo Search")
ps[3].runs[0].font.size = Pt(40)
for r in ps[5].runs:
    r.text = ""
names = shape(s, "66;p1")
names.top = Inches(5.7)
nb = names.text_frame._txBody
nps = nb.findall(f"{A}p")
proto = nps[2]
for p in nps:
    nb.remove(p)
for ln in ["Abhijith M Reddy (1BM23CD003)", "Aniketh S Nadig (1BM23CD006)", "Mayur M (1BM23CD037)",
           "S Sanjay (1BM23CD053)", "Tanush V Gowda (1BM23CD064)", "", "Prof. Manoj Kumar",
           "Assistant Professor", "Department of CSE(Data Science)", "BMSCE, Bengaluru"]:
    nb.append(_with_text(proto, ln, 30))

# ================================================================== 2 Table of contents
toc_box = shape(S[1], "81;p2")
tb = toc_box.text_frame._txBody
tps = tb.findall(f"{A}p")
proto = tps[0]
for p in tps:
    tb.remove(p)
for item in ["Introduction to optimization problems", "Nature inspired algorithms", "Problem statement",
             "Dataset and preprocessing", "Implementation", "Performance metrics and results",
             "Advantages and limitations", "Conclusion"]:
    tb.append(_with_text(proto, item, 44))

# ================================================================== 3 Introduction
s = S[2]
set_title(s, "1. Introduction to Optimization Problems")
fill(body(s), [
    ("h", "What is Optimization?"),
    ("s", "Finding the best solution from all possible solutions"),
    ("s", "Minimizing or maximizing an objective function"),
    ("s", "Subject to constraints and limitations"),
    ("b",),
    ("h", "Hyperparameter Tuning as Optimization:"),
    ("s", "Objective: validation error of the trained model"),
    ("s", "No gradient: every evaluation trains and cross-validates a model"),
    ("s", "Noisy, non-convex search space with many local optima"),
    ("b",),
    ("h", "Types of Optimization Problems:"),
    ("s", "Continuous: real-valued parameters (e.g., C, gamma, epsilon of an SVR)"),
    ("s", "Binary: whether to use an input feature or not"),
    ("s", "Mixed: both together (our problem: 3 continuous + 18 binary variables)"),
])

# ================================================================== 4 Nature inspired algorithms
s = S[3]
set_title(s, "2. Nature Inspired Algorithms")
fill(body(s), [
    ("h", "Definition:"),
    ("s", "Optimization algorithms inspired by natural phenomena"),
    ("s", "Population-based, gradient-free metaheuristics"),
    ("b",),
    ("h", "Popular Nature-Inspired Algorithms:"),
    ("s", "Genetic Algorithm (GA): evolution and natural selection"),
    ("s", "Particle Swarm Optimization (PSO): bird flocking behavior"),
    ("s", "Ant Colony Optimization (ACO): ant foraging patterns"),
    ("s", "Differential Evolution (DE): vector-difference mutation"),
    ("s", "Cuckoo Search (CS): brood parasitism of cuckoos with Lévy flights"),
    ("b",),
    ("h", "Why Cuckoo Search?"),
    ("s", "Only two parameters to set: discovery probability pa and step size α"),
    ("s", "Lévy flights make occasional long jumps across log-scale ranges"),
    ("s", "Nest abandonment keeps the population diverse"),
], s=31)

# ================================================================== 5 Problem statement
s = S[4]
set_title(s, "3. Problem Statement")
fill(body(s), [
    ("h", "Research Challenge:"),
    ("s", "Parkinson's severity (UPDRS) is scored by a clinician during a clinic visit"),
    ("s", "Voice changes early in Parkinson's and can be recorded at home"),
    ("s", "SVR accuracy depends heavily on C, gamma, epsilon and the input features"),
    ("b",),
    ("h", "Specific Problem:"),
    ("s", "Predict the motor UPDRS score from 16 voice measures + age and sex"),
    ("s", "Jointly tune SVR hyperparameters and select features with Cuckoo Search"),
    ("s", "Beat the default SVR (R² = 0.50) and Random Search with the same budget"),
    ("b",),
    ("h", "Target Variable:"),
    ("s", "Motor UPDRS score, range 5 – 40 (mean 21.3)"),
])

# ================================================================== 6 Dataset
s = S[5]
set_title(s, "4. Dataset and Its Preprocessing")
fill(body(s), [
    ("h", "Dataset:"),
    ("s", "UCI Parkinson's Telemonitoring (Oxford + Intel, six-month home trial)"),
    ("s", "5,875 voice recordings from 42 patients"),
    ("s", "18 features: age, sex, 5 jitter, 6 shimmer, NHR, HNR, RPDE, DFA, PPE"),
    ("s", "Target: motor UPDRS score"),
    ("b",),
    ("h", "Preprocessing:"),
    ("s", "No missing values or duplicate rows"),
    ("s", "log(1+x) transform on 12 skewed jitter, shimmer and NHR features"),
    ("s", "80/20 train/test split: 4,700 / 1,175 recordings"),
    ("s", "Features and target standardised inside the model pipeline"),
    ("s", "Test set never used during the search"),
])

# ================================================================== 7 Implementation (two columns, sample slide 7)
s = S[6]
set_title(s, "5. Implementation")
fill(shape(s, "149;p7"), [
    ("h", "System Architecture:"),
    ("s", "Data cleaning and preprocessing"),
    ("s", "RBF-SVR model (scikit-learn)"),
    ("s", "Cuckoo Search written from scratch"),
    ("s", "Fitness: 3-fold CV RMSE (training set)"),
    ("s", "Random Search with the same budget"),
    ("b",),
    ("h", "Search Space (21 dimensions):"),
    ("s", "C: [0.1 – 1000] (log scale)"),
    ("s", "Gamma: [0.0001 – 10] (log scale)"),
    ("s", "Epsilon: [0.001 – 1] (log scale)"),
    ("s", "18 feature bits (selected if > 0.5)"),
], h=32, s=29)
fill(shape(s, "151;p7"), [
    ("h", "Cuckoo Search Configuration:"),
    ("s", "Nests: 20"),
    ("s", "Generations: 20 (≈ 820 evaluations)"),
    ("s", "Discovery probability pa: 0.25"),
    ("s", "Lévy exponent β: 1.5"),
    ("b",),
    ("h", "Operators:"),
    ("s", "Lévy flight egg laying"),
    ("s", "Greedy nest replacement"),
    ("s", "Nest abandonment"),
    ("s", "Elitism and warm start"),
    ("b",),
    ("h", "Validation:"),
    ("s", "Beats Random Search on 4 benchmark functions"),
], h=32, s=29)

# ================================================================== 8 Performance metrics (table)
s = S[7]
set_title(s, "6. Performance Metrics")
rows = [["Model (test set)", "RMSE", "MAE", "R²", "MAPE"],
        ["Mean predictor", "7.991", "6.878", "0.000", "43.1%"],
        ["Linear Regression", "7.467", "6.327", "0.126", "38.4%"],
        ["Default SVR (no optimization)", "5.672", "4.411", "0.496", "26.5%"],
        ["Random Search SVR", "4.693", "3.297", "0.655", "19.5%"],
        ["Cuckoo-SVR (proposed)", "4.347", "3.066", "0.704", "18.0%"]]
gf = s.shapes.add_table(len(rows), 5, Inches(2.0), Inches(1.75), Inches(18.0), Inches(0.68 * len(rows)))
gf.name = "Results table"
tbl = gf.table
for j, w in enumerate([7.6, 2.6, 2.6, 2.6, 2.6]):
    tbl.columns[j].width = Inches(w)
for i, row in enumerate(rows):
    tbl.rows[i].height = Inches(0.68)
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.text = val
        par = c.text_frame.paragraphs[0]
        par.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        f = par.runs[0].font
        f.name, f.size, f.bold = FONT, Pt(30), i in (0, len(rows) - 1)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
b = body(s)
b.top, b.height = Inches(6.0), Inches(4.9)
fill(b, [
    ("h", "Evaluation Metrics Used: RMSE, MAE, R², MAPE"),
    ("b",),
    ("h", "Default SVR → Cuckoo-SVR:"),
    ("s", "RMSE 23.4% lower, MAE 30.5% lower, MAPE 32.2% lower"),
    ("s", "R² improved from 0.50 to 0.70, using only 9 of 18 features"),
    ("s", "Also beats Random Search on every metric (about 7%)"),
], h=34, s=31)

# ================================================================== 9 Convergence and robustness (one figure)
s = S[8]
set_title(s, "6. Convergence and Robustness")
iw, ih = Image.open(FIG + "convergence.png").size
pw = 15.5
pic = s.shapes.add_picture(FIG + "convergence.png", Inches((21.99 - pw) / 2), Inches(1.55), Inches(pw), Inches(pw * ih / iw))
pic.name = "Convergence plot"
b = body(s)
b.top, b.height = Inches(1.6 + pw * ih / iw + 0.1), Inches(3.6)
fill(b, [
    ("h", "Convergence: Random Search stalls at 4.780; Cuckoo Search keeps improving to 4.426"),
    ("h", "Robustness over 10 runs: Cuckoo 4.356 ± 0.042 vs Random Search 4.596 ± 0.186"),
    ("s", "Beat the default SVR in 10/10 runs (Wilcoxon p = 0.001)"),
    ("s", "Beat Random Search in 8/10 runs (Wilcoxon p = 0.005)"),
], h=32, s=30)

# ================================================================== 10 Advantages
s = S[9]
set_title(s, "7. Advantages")
fill(body(s), [
    ("h", "Advantages:"),
    ("s", "Superior Accuracy: 23.4% lower RMSE than the default SVR"),
    ("s", "Beats Random Search with the same budget (8/10 runs, p = 0.005)"),
    ("s", "Robustness: low variance across 10 runs (std 0.042 vs 0.186)"),
    ("s", "Feature Selection: automatically reduced 18 features to 9"),
    ("s", "Simplicity: only two algorithm parameters (pa and α)"),
    ("s", "Validated: outperforms Random Search on benchmark functions"),
    ("s", "Practical: runs on a laptop CPU; fully reproducible notebook"),
])

# ================================================================== 11 Limitations
s = S[10]
set_title(s, "7. Limitations")
fill(body(s), [
    ("h", "Limitations:"),
    ("s", "Random split: the same patients appear in train and test sets"),
    ("s", "Suited to monitoring known patients, not diagnosing new ones"),
    ("s", "Computational Cost: about 3 hours for the full 10-run study"),
    ("s", "Single Objective: optimizes RMSE only"),
    ("s", "Preprocessing: log(1+x) barely reduced the skew of small-valued features"),
    ("s", "Label Noise: UPDRS interpolated between three clinic visits"),
])

# ================================================================== 12 Conclusion
s = S[11]
set_title(s, "8. Conclusion")
fill(body(s), [
    ("h", "Conclusion:"),
    ("s", "Cuckoo Search implemented from scratch and validated on benchmark functions"),
    ("s", "Test RMSE 5.672 → 4.347 (23.4% lower); R² 0.496 → 0.704"),
    ("s", "Only 9 of the 18 voice features were needed"),
    ("s", "Significantly better than Random Search with the same budget (p = 0.005)"),
    ("b",),
    ("h", "Future Work:"),
    ("s", "Patient-wise validation on a larger group of patients"),
    ("s", "Multi-objective search: accuracy vs number of features"),
    ("s", "Comparison with GA, PSO and DE"),
])

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
