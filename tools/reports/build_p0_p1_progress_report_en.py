from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/p0_p1_ordered_1d_empirical_mh_progress_en.pdf"
P0 = json.loads((ROOT / "results/mtj_p0_direct_pdit_output/p0_summary.json").read_text())
FOUR = json.loads((ROOT / "results/mtj_p0_four_solver_output/four_solver_summary.json").read_text())
P1 = json.loads((ROOT / "results/mtj_p1_empirical_transition_mh_output/p1_summary.json").read_text())
TSP = json.loads(
    (ROOT / "results/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh/tsp_comparison_summary.json").read_text()
)

PAGE_W, PAGE_H = A4
MX, MT, MB = 17 * mm, 17 * mm, 16 * mm
CW = PAGE_W - 2 * MX

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2B6F9F")
ORANGE = colors.HexColor("#D9772D")
GREEN = colors.HexColor("#2F9E62")
GRAY = colors.HexColor("#667085")
DARK = colors.HexColor("#1C2430")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_ORANGE = colors.HexColor("#FCEFE4")
LIGHT_GREEN = colors.HexColor("#E9F6EF")
LIGHT_GRAY = colors.HexColor("#F3F5F7")


def style(name: str, **kw) -> ParagraphStyle:
    base = dict(
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.8,
        textColor=DARK,
        spaceAfter=5,
    )
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "cover_kicker": style("cover_kicker", fontSize=13, leading=18, textColor=BLUE),
    "cover_title": style(
        "cover_title", fontName="Helvetica-Bold", fontSize=24, leading=31, textColor=NAVY
    ),
    "h1": style("h1", fontName="Helvetica-Bold", fontSize=18, leading=24, textColor=NAVY),
    "h2": style(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=17,
        textColor=BLUE,
        spaceBefore=5,
    ),
    "body": style("body"),
    "small": style("small", fontSize=7.6, leading=10.5, textColor=GRAY),
    "caption": style(
        "caption",
        fontSize=7.5,
        leading=10.3,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceBefore=3,
    ),
    "callout": style("callout", fontSize=10, leading=15.5, textColor=NAVY),
    "table": style("table", fontSize=7.3, leading=9.8, spaceAfter=0),
    "table_head": style(
        "table_head",
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.5,
        textColor=colors.white,
        alignment=TA_CENTER,
        spaceAfter=0,
    ),
    "metric": style(
        "metric",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=0,
    ),
    "metric_label": style(
        "metric_label",
        fontSize=7.2,
        leading=9.2,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceAfter=0,
    ),
}


def p(text: str, kind: str = "body") -> Paragraph:
    return Paragraph(text, S[kind])


def bullet(text: str) -> Paragraph:
    return Paragraph("- " + text, S["body"])


def callout(text: str, background=LIGHT_BLUE) -> Table:
    t = Table([[p(text, "callout")]], colWidths=[CW])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#C9D8E5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return t


def metric_cards(items: list[tuple[str, str]], background=LIGHT_BLUE) -> Table:
    cards = []
    iw = CW / len(items) - 8
    for value, label in items:
        inner = Table([[p(value, "metric")], [p(label, "metric_label")]], colWidths=[iw])
        inner.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        cards.append(inner)
    t = Table([cards], colWidths=[CW / len(items)] * len(items))
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8E3")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def table(rows: list[list[str]], widths: list[float]) -> Table:
    cells = []
    for i, row in enumerate(rows):
        kind = "table_head" if i == 0 else "table"
        cells.append([p(str(x), kind) for x in row])
    t = Table(cells, colWidths=widths, repeatRows=1)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D0D9")),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    t.setStyle(TableStyle(commands))
    return t


def figure(path: str, max_h: float) -> list:
    full = ROOT / path
    iw, ih = ImageReader(str(full)).getSize()
    scale = min(CW / iw, max_h / ih)
    return [Image(str(full), width=iw * scale, height=ih * scale)]


def fig(path: str, caption: str, max_h: float) -> list:
    return figure(path, max_h) + [p(caption, "caption")]


def decorate(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D7DEE7"))
        canvas.line(MX, PAGE_H - 12 * mm, PAGE_W - MX, PAGE_H - 12 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(MX, PAGE_H - 9 * mm, "sMTJ P-dit Progress Report")
        canvas.drawRightString(PAGE_W - MX, PAGE_H - 9 * mm, "Ordered 1D Empirical-Transition MH")
    canvas.setStrokeColor(colors.HexColor("#D7DEE7"))
    canvas.line(MX, 11 * mm, PAGE_W - MX, 11 * mm)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(GRAY)
    canvas.drawString(MX, 7 * mm, "Results from saved repository outputs - 2026-07-24")
    canvas.drawRightString(PAGE_W - MX, 7 * mm, str(doc.page))
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=MX,
    rightMargin=MX,
    topMargin=MT,
    bottomMargin=MB,
    title="P0-P1 Progress: Ordered 1D Empirical-Transition MH for TSP",
    author="sMTJ annealing accelerator project",
    subject="P-dit progress and end-to-end TSP comparison",
)
frame = Frame(MX, MB, CW, PAGE_H - MT - MB, 0, 0, 0, 0)
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])

story = []

# 1 - Cover
story += [
    Spacer(1, 20 * mm),
    p("P0 TO P1 PROGRESS", "cover_kicker"),
    p("Ordered 1D Empirical-Transition MH<br/>for Direct-Permutation TSP Optimization", "cover_title"),
    Spacer(1, 5 * mm),
    callout(
        "<b>This report uses an end-to-end comparison.</b> The binary p-bit baseline follows the "
        "one-hot QUBO/Ising route. The P-dit method uses the new direct-permutation encoding and "
        "the ordered 1D empirical-transition MH kernel. The objective is to compare the complete "
        "implementable methods, not to keep the original encoding fixed.",
        LIGHT_ORANGE,
    ),
    Spacer(1, 9 * mm),
    metric_cards(
        [
            ("100%", "ordered-route validity"),
            ("24/24", "8-city exact hits"),
            ("13/24", "12-city reference hits"),
            ("1.37e-16", "max relative DB error"),
        ],
        LIGHT_GREEN,
    ),
    Spacer(1, 12 * mm),
    p(
        "Scope: P0 direct categorical P-dit, measured sMTJ transition-driven MH, P1 state-geometry "
        "analysis, and the new kernel on 8-city and 12-city TSP instances relative to binary p-bits."
    ),
    Spacer(1, 4 * mm),
    p("Dataset: selected_curve | K=6 | Report date: 2026-07-24", "small"),
    PageBreak(),
]

# 2 - Executive summary
story += [
    p("1. Executive Summary", "h1"),
    callout(
        "The decisive TSP improvement is not obtained by replacing the random source inside the "
        "same one-hot Ising model. It comes from changing both the <b>state representation</b> and "
        "the <b>update kernel</b>: the route is stored directly as a permutation, while a measured "
        "device transition k-to-l proposes a local transposition through "
        "q<sub>kl</sub>=r<sub>k</sub>P<sub>kl</sub>, followed by a Hastings correction."
    ),
    Spacer(1, 5 * mm),
    p("Development path", "h2"),
    table(
        [
            ["Stage", "Problem addressed", "Main output"],
            ["P0-A", "Remove invalid Gray/one-hot states and penalties", "Direct categorical P-dit; every represented state is valid"],
            ["P0-B", "Use measured device dynamics as proposals", "q<sub>kl</sub>=r<sub>k</sub>P<sub>kl</sub> plus MH correction"],
            ["P1", "Identify the geometry supported by the K=6 device", "Ordered 1D locality=0.708 versus Gray locality=0.363"],
            ["TSP", "Eliminate the one-hot feasibility bottleneck", "Direct permutation plus reversible cyclic transpositions"],
        ],
        [27 * mm, 69 * mm, 75 * mm],
    ),
    Spacer(1, 7 * mm),
    p("Interpretation of the binary p-bit comparison", "h2"),
    bullet("<b>Main comparison:</b> binary p-bit/one-hot Ising versus ordered 1D direct-permutation MH."),
    bullet(
        "<b>Not the main result:</b> the fixed-one-hot P1 ablation that only changes the Gray/1D "
        "embedding. That experiment remains a diagnostic showing that a random-primitive swap alone "
        "does not remove the TSP representation bottleneck."
    ),
    bullet(
        "<b>Attribution boundary:</b> the observed gain combines encoding compression, feasible-space "
        "reconstruction, proposal locality, and the MH kernel. It cannot be assigned to geometry alone."
    ),
    Spacer(1, 7 * mm),
    callout(
        "The current ordered 1D empirical-transition MH path closes the loop between device physics, "
        "state geometry, and TSP constraints: the K=6 state is an actual proposal state, and every "
        "route update stays inside the legal permutation space.",
        LIGHT_GREEN,
    ),
    PageBreak(),
]

# 3 - P0 direct P-dit
enc = P0["encoding_comparison"]
story += [
    p("2. P0: From Encoding Overhead to Direct Categorical P-dits", "h1"),
    p(
        "P0 first established that direct multivalued variables can replace Gray or one-hot expansions. "
        "In the 14-item, 3-partition validation problem, each item stores one category "
        "z<sub>i</sub> in {0,1,2} and is updated through a softmax conditional distribution. "
        "No invalid-code handling or one-hot penalty is required."
    ),
    metric_cards(
        [
            (str(enc["direct_pdit_variables"]), "direct P-dit variables"),
            (str(enc["gray_binary_variables"]), "Gray binary variables"),
            (str(enc["onehot_binary_variables"]), "one-hot binary variables"),
            (f'{100*P0["final"]["ground_hit_rate"]:.0f}%', "ground hit at 2048 clocks"),
        ]
    ),
    Spacer(1, 6 * mm),
    *fig(
        "results/mtj_p0_direct_pdit_output/pdit_partition_performance.png",
        "Figure 1. P0 direct P-dit partition solver. The ground-hit rate reaches 90% by 1024 "
        "physical-clock iterations; the median best error is already zero at 512 iterations.",
        88 * mm,
    ),
    p("Physical-data processing correction completed in P0", "h2"),
    bullet(
        "Duration-proportional integer apportionment preserves all 135 true boundaries across 136 dwell segments."
    ),
    bullet(
        f'The new resampling method produces {P0["resampling_validation"]["new_phantom_transition_count"]} '
        "phantom transitions, avoiding artifacts introduced by uniform index resampling."
    ),
    bullet(
        "This stage proved that direct multivalued states remove invalid encoding regions, but it did "
        "not yet determine the correct geometry for the measured K=6 states."
    ),
    PageBreak(),
]

# 4 - P0 empirical transition
four = {x["solver"]: x for x in FOUR["summaries"]}
story += [
    p("3. P0: From Direct P-dits to Measured-Transition MH", "h1"),
    p(
        "A four-solver experiment then separated representation effects from physical-proposal effects. "
        "Solver D introduced a direct categorical proposal based on the measured "
        "q<sub>kl</sub>=r<sub>k</sub>P<sub>kl</sub>, with a Metropolis-Hastings correction for asymmetric "
        "forward and reverse rates."
    ),
    *fig(
        "results/mtj_p0_four_solver_output/four_solver_success.png",
        "Figure 2. P0 four-solver success curves. C is the software direct P-dit; D is the physical "
        "empirical-transition MH P-dit; A and B are Gray-binary paths. This validates the design path "
        "but is not the final TSP performance comparison.",
        86 * mm,
    ),
]
rows = [["Solver", "Success at 4000 updates", "Median first hit", "Valid-state rate"]]
for key in ["physical_gray_pbit", "software_gray_glauber", "direct_pdit", "physical_mh_pdit"]:
    x = four[key]
    rows.append(
        [
            x["label"],
            f'{100*x["success_probability"]:.1f}%',
            f'{x["median_first_hit_update_successes"]:.1f}',
            f'{100*x["valid_state_fraction"]:.0f}%',
        ]
    )
story += [
    table(rows, [64 * mm, 38 * mm, 38 * mm, 31 * mm]),
    Spacer(1, 5 * mm),
    p("Design constraints established at this stage", "h2"),
    bullet(
        "The MH acceptance ratio must include q<sub>lk</sub>/q<sub>kl</sub>; otherwise, asymmetric "
        "measured device transitions alter the target stationary distribution."
    ),
    bullet(
        "A direct categorical representation guarantees validity, but the proposal geometry must still "
        "match the device's dominant transition directions."
    ),
    bullet(
        "P1 therefore focused on identifying which geometry is actually supported by the K=6 transition graph."
    ),
    PageBreak(),
]

# 5 - P1 geometry
geo = {x["geometry"]: x for x in P1["geometry_compatibility"]}
story += [
    p("4. P1: The Measured K=6 Transition Graph Favors Ordered 1D", "h1"),
    p(
        "The six selected_curve states are naturally ordered by their resistance/conductance means. "
        "P1 compared the measured proposal-rate matrix with Gray-coded binary, ordered 1D, and isotropic "
        "six-state geometries. Strong transitions concentrate between nearby ordered states rather than "
        "forming an isotropic all-to-all graph."
    ),
    *fig(
        "results/mtj_p1_empirical_transition_mh_output/p1_transition_geometry.png",
        "Figure 3. Left: measured q matrix. Remaining panels: relative affinity under each geometry. "
        "The banded ordered-1D structure better matches the measured strong transitions; the isotropic "
        "model treats every off-diagonal pair as equally distant.",
        58 * mm,
    ),
    table(
        [
            ["Geometry", "Proposal locality", "q-weighted normalized distance", "Rate vs -distance Pearson"],
            [
                "Gray binary",
                f'{geo["gray_binary"]["proposal_locality_score"]:.3f}',
                f'{geo["gray_binary"]["q_weighted_mean_normalized_distance"]:.3f}',
                f'{geo["gray_binary"]["symmetric_rate_vs_negative_distance_pearson"]:.3f}',
            ],
            [
                "<b>Ordered 1D</b>",
                f'<b>{geo["ordered_1d"]["proposal_locality_score"]:.3f}</b>',
                f'<b>{geo["ordered_1d"]["q_weighted_mean_normalized_distance"]:.3f}</b>',
                f'<b>{geo["ordered_1d"]["symmetric_rate_vs_negative_distance_pearson"]:.3f}</b>',
            ],
            ["Isotropic six-state", "0.000", "1.000", "Undefined: all distances are equal"],
        ],
        [42 * mm, 38 * mm, 50 * mm, 41 * mm],
    ),
    Spacer(1, 5 * mm),
    callout(
        "Why could the isotropic P-dit avoid one-hot encoding while the earlier 1D path could not? "
        "The limitation was not one-dimensionality itself. The earlier implementation retained N-squared "
        "one-hot TSP variables. The new direct-permutation mapping removes one-hot encoding while preserving "
        "the device's measured local ordering.",
        LIGHT_ORANGE,
    ),
    PageBreak(),
]

# 6 - P1 sampling
p1sum = {x["geometry"]: x for x in P1["summaries"]}
story += [
    p("5. P1: Empirical-Transition MH Sampling Validation", "h1"),
    p(
        "The enumerated validation problem uses four variables, six states, 64 traces per geometry, and "
        "8000 updates per trace. All geometries use the same q<sub>kl</sub> proposal and MH framework; only "
        "the energy geometry changes. Ordered 1D achieves both higher event acceptance and shorter energy "
        "autocorrelation."
    ),
    *fig(
        "results/mtj_p1_empirical_transition_mh_output/p1_sampling_metrics.png",
        "Figure 4. P1 sampling metrics. Ordered 1D reaches an MH acceptance per event of about 0.490 "
        "and a median energy IAT of about 52.3 updates. Gray reaches about 0.219 and 121.3, respectively.",
        108 * mm,
    ),
]
rows = [["Geometry", "Stationary KL", "MH acceptance/event", "Median IAT", "Success"]]
for key in ["gray_binary", "ordered_1d", "isotropic_6state"]:
    x = p1sum[key]
    rows.append(
        [
            x["label"],
            f'{x["stationary_kl_empirical_to_exact"]:.4f}',
            f'{x["mh_acceptance_given_event"]:.3f}',
            f'{x["median_energy_iat_updates"]:.1f}',
            f'{100*x["success_probability"]:.1f}%',
        ]
    )
story += [
    table(rows, [48 * mm, 31 * mm, 38 * mm, 29 * mm, 25 * mm]),
    Spacer(1, 5 * mm),
    p(
        "Ordered 1D is not guaranteed to be fastest for every objective. It does, however, use the current "
        "K=6 device's local transition structure most faithfully. This supports mapping device transition "
        "distance to route-transposition distance in TSP."
    ),
    PageBreak(),
]

# 7 - Kernel
validation = TSP["kernel_validation"]
story += [
    p("6. TSP Kernel: Direct Permutations plus Ordered 1D Empirical-Transition MH", "h1"),
    p("State and proposal", "h2"),
    callout(
        "Route state: pi=[0, pi<sub>1</sub>, ..., pi<sub>N-1</sub>], with city 0 fixed to remove rotational "
        "degeneracy. A device event k-to-l occurs at q<sub>kl</sub>=r<sub>k</sub>P<sub>kl</sub>. "
        "|l-k| is mapped to a cyclic transposition distance outside the fixed origin."
    ),
    Spacer(1, 5 * mm),
    p("Acceptance rule", "h2"),
    callout(
        "A[(pi,k) to (pi',l)] = min{1, exp[-beta(L(pi')-L(pi))] * q<sub>lk</sub>/q<sub>kl</sub>}",
        LIGHT_ORANGE,
    ),
    Spacer(1, 5 * mm),
    p("How the kernel eliminates the one-hot bottleneck", "h2"),
    table(
        [
            ["Property", "Binary p-bit / one-hot Ising", "New ordered 1D permutation MH"],
            ["State size", "N-squared binary variables", "N route entries; only (N-1)! fixed-origin routes"],
            ["Validity", "Approximated by row/column penalties", "Every transposition remains a permutation"],
            ["Device role", "Random update or bias source", "k-to-l is the actual q<sub>kl</sub> proposal state"],
            ["Reversibility", "Defined by the Ising update rule", "Same swap for an unordered state pair; MH corrects q asymmetry"],
        ],
        [30 * mm, 67 * mm, 74 * mm],
    ),
    Spacer(1, 7 * mm),
    p("Kernel validation", "h2"),
    metric_cards(
        [
            ("630", "detailed-balance flux cases"),
            (f'{validation["theoretical_db_max_relative_flux_error"]:.2e}', "max relative flux error"),
            ("PASS", "route move reversible under k/l swap"),
            ("PASS", "adjacent device edges bidirectional"),
        ],
        LIGHT_GREEN,
    ),
    Spacer(1, 6 * mm),
    bullet(
        "Distance-1 swaps have positive probability, and adjacent transpositions generate every fixed-origin permutation."
    ),
    bullet(
        "An accepted move updates both route and device state; a rejection leaves both unchanged, preserving "
        "detailed balance in the joint chain."
    ),
    PageBreak(),
]

# 8 - TSP results
tsp = {(x["n_cities"], x["method"]): x for x in TSP["method_summary"]}
story += [
    p("7. End-to-End TSP Performance: New Encoding and Kernel vs Binary P-bits", "h1"),
    p(
        "This is the requested complete-method comparison, not a one-factor ablation. The binary baseline "
        "uses one-hot QUBO/Ising. The P-dit method simultaneously uses direct-permutation encoding and ordered "
        "1D empirical-transition MH. Each method has 24 runs per city size. The saved experiment happens to "
        "use 121,500 algorithmic update attempts, but the meaning of an update differs between the two paths."
    ),
    *fig(
        "results/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh/tsp_method_comparison.png",
        "Figure 5. Main TSP comparison. Left: route quality before 2-opt; center: raw-assignment validity; "
        "right: representation size. The binary baseline is labeled 'Binary p-bit + one-hot Ising'.",
        67 * mm,
    ),
]
rows = [["Cities", "Method", "Variables", "Raw feasible", "Median route/ref", "Raw reference hit"]]
for n in [8, 12]:
    for method, label in [
        ("binary_pbit_onehot_ising", "Binary p-bit / one-hot"),
        ("ordered_1d_empirical_mh_pdit", "Ordered 1D permutation MH"),
    ]:
        x = tsp[(n, method)]
        rows.append(
            [
                str(n),
                label,
                str(x["variables"]),
                f'{100*x["raw_feasible_rate"]:.1f}%',
                f'{x["median_route_ratio"]:.3f}',
                f'{100*x["raw_reference_hit_rate"]:.1f}%',
            ]
        )
story += [
    table(rows, [16 * mm, 53 * mm, 19 * mm, 28 * mm, 30 * mm, 30 * mm]),
    Spacer(1, 5 * mm),
    p("Key readings", "h2"),
    bullet(
        "<b>8 cities:</b> ordered MH uses 8 variables and reaches the exact reference in 24/24 raw runs. "
        "The binary one-hot path uses 64 variables, has zero raw reference hits, and a median route/ref of 1.263."
    ),
    bullet(
        "<b>12 cities:</b> ordered MH uses 12 variables, maintains 100% raw feasibility, and reaches the "
        "reference in 13/24 runs. The binary one-hot path uses 144 variables, has 0% raw feasibility, "
        "and a median one-hot violation of 10.5."
    ),
    bullet(
        "The 12-city reference is obtained by nearest-neighbor all-start plus 2-opt and is not guaranteed "
        "to be globally optimal. A ratio of 1 means that this reference was reached."
    ),
    PageBreak(),
]

# 9 - Route example and limitations
story += [
    p("8. Route Example, Sources of Improvement, and Boundaries", "h1"),
    *fig(
        "results/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh/tsp_best_routes_comparison.png",
        "Figure 6. Best saved 12-city routes. Left: decoded binary one-hot route, length 51.581. "
        "Right: ordered 1D empirical-MH route, length 34.645, reaching the current reference.",
        80 * mm,
    ),
    p("Three layers contribute to the gain", "h2"),
    table(
        [
            ["Layer", "Mechanism", "Observed effect"],
            ["Representation", "N-squared one-hot to N-entry permutation", "8x/12x fewer variables for 8/12 cities; search remains legal"],
            ["Geometry", "|l-k| controls transposition distance", "Accepted mean distance is about 1.10, preserving local search"],
            ["Dynamics", "Measured q<sub>kl</sub> proposal plus Hastings correction", "Keeps device asymmetry while preserving target detailed balance"],
        ],
        [27 * mm, 65 * mm, 79 * mm],
    ),
    Spacer(1, 6 * mm),
    p("Limitations and correct interpretation", "h2"),
    bullet(
        "This is an end-to-end algorithm comparison, not a hardware-latency or energy-efficiency comparison. "
        "Python wall time does not predict future sMTJ-array speed."
    ),
    bullet(
        "Update semantics differ: the one-hot path updates one Ising spin; the ordered path advances one "
        "empirical proposal-clock tick. Software runtime alone should not be used to rank device performance."
    ),
    bullet(
        "There are 24 runs per cell. The 8-city reference is exact; the 12-city reference is heuristic. "
        "Larger-scale claims require more instances, random seeds, and confidence intervals."
    ),
    bullet(
        "The P1 fixed-encoding experiment remains useful for geometry diagnosis, but it is not used for the "
        "TSP performance claim in this report."
    ),
    PageBreak(),
]

# 10 - Conclusions
story += [
    p("9. Current Conclusion and Next Steps", "h1"),
    callout(
        "<b>Current conclusion:</b> ordered 1D empirical-transition MH now forms a complete path from "
        "measured K=6 transition statistics to TSP route proposals. Direct-permutation encoding removes the "
        "one-hot feasibility bottleneck structurally, while measured-q proposals and the Hastings correction "
        "provide both device consistency and theoretical reversibility.",
        LIGHT_GREEN,
    ),
    Spacer(1, 7 * mm),
    p("Recommended validation priorities", "h2"),
    table(
        [
            ["Priority", "Work", "Purpose"],
            ["1", "Extend to 16/20/30 cities, multiple instances, and multiple seeds", "Test whether the representation advantage remains stable with scale"],
            ["2", "Report success vs clock, proposal events, and accepted swaps", "Prevent ambiguity from different update semantics"],
            ["3", "Add adaptive beta and a distance-aware proposal mixture", "Balance local refinement and long-range escape"],
            ["4", "Build parallel device mapping plus latency/energy models", "Translate algorithmic gains into hardware metrics"],
        ],
        [18 * mm, 84 * mm, 69 * mm],
    ),
    Spacer(1, 8 * mm),
    p("Reproduction and figure sources", "h2"),
    p(
        "P0: src/mtj_p0_direct_pdit_partition.py, src/mtj_p0_four_solver_benchmark.py<br/>"
        "P1: src/mtj_p1_empirical_transition_mh.py<br/>"
        "TSP: src/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh.py<br/>"
        "Figures 1-6 are embedded directly from the PNG files saved by these scripts under results/. "
        "No plot data were redrawn or modified."
    ),
    Spacer(1, 5 * mm),
    p("Reference", "h2"),
    p(
        "P-dits: Probabilistic d-dimensional bits for extended-variable probabilistic computing. "
        "DOI: 10.1103/4ngx-cmz7",
        "small",
    ),
    Spacer(1, 7 * mm),
    callout(
        "<b>One-sentence summary:</b> the TSP advance comes from the combination of ordered 1D device "
        "geometry, empirical-transition MH, and permutation-preserving encoding - not from replacing the "
        "random source inside the original one-hot model.",
        LIGHT_ORANGE,
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story)
print(OUT)
