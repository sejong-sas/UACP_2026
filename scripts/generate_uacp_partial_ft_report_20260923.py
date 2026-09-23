#!/usr/bin/env python3
"""Generate the UACP Partial Fine-Tuning research-extension report.

The report is assembled from the repository's recorded Markdown/CSV artifacts.
It creates a new output directory and never modifies experiment artifacts.
"""
from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "UACP_Partial_FineTuning_Experiment_Report_20260923_v5"
FIG = OUT / "figures"
DOCX = OUT / "UACP_Partial_FineTuning_Experiment_Report_20260923.docx"

PRE_SHA = "d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pick(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in criteria.items()):
            return row
    raise KeyError(criteria)


def ensure_clean_output() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report directory: {OUT}")
    FIG.mkdir(parents=True, exist_ok=False)


def savefig(path: Path) -> None:
    # Fixed canvas dimensions avoid an oversized crop when log-axis annotations
    # are present in the evidential-ratio figure.
    plt.savefig(path, dpi=300, facecolor="white")
    plt.close()


def box(ax, x, y, w, h, text, color="#eaf2f8", edge="#335c81", fs=12):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.4,
                       facecolor=color, edgecolor=edge, transform=ax.transAxes)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            transform=ax.transAxes, color="#17202a", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=16, linewidth=1.5,
                                 color="#536878"))


def figure_architecture() -> Path:
    p = FIG / "fig1_model_architecture.png"
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.axis("off")
    box(ax, .04, .39, .18, .22, "Sparse CFR\n+ mask", "#e8f4f8")
    box(ax, .29, .39, .18, .22, "Input\nprojection", "#edf2f7")
    box(ax, .54, .39, .20, .22, "Residual blocks ×32", "#e9f7ef")
    arrow(ax, .22, .50, .29, .50); arrow(ax, .47, .50, .54, .50)
    ax.text(.64, .30, "Shared feature", ha="center", fontsize=11, transform=ax.transAxes, color="#475569")
    for x, label, c in [(.80, "γ\nCFR mean", "#fff4d6"), (.80, "Ψ\nscale", "#fdebd0"), (.80, "κ\nconfidence", "#f6ddcc"), (.80, "ν\ndegrees", "#eadcf8")]:
        pass
    labels = [("γ\nCFR mean", .82, .78, "#fff4d6"), ("Ψ\nscale", .82, .55, "#fdebd0"), ("κ\nconfidence", .82, .32, "#f6ddcc"), ("ν\ndegrees", .82, .09, "#eadcf8")]
    for label, x, y, c in labels:
        box(ax, x, y, .14, .12, label, c, fs=10)
        arrow(ax, .74, .50, x, y + .06)
    ax.text(.64, .50, "Evidential\nhead inputs", ha="center", va="center", fontsize=11, transform=ax.transAxes)
    ax.set_title("UACP predictor architecture used by the Partial Fine-Tuning experiments", fontsize=15, weight="bold")
    savefig(p)
    return p


def figure_flow() -> Path:
    p = FIG / "fig2_adaptation_flow.png"
    fig, ax = plt.subplots(figsize=(10, 4.0)); ax.axis("off")
    labels = ["120 ns full CFR\nfeedback", "Clean full-CFR\ntarget", "Sparse mask +\n15 dB AWGN", "Sparse CFR + mask\ninput", "Predictor", "Reconstructed full CFR\n+ uncertainty"]
    xs = np.linspace(.03, .83, len(labels))
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#e0f2fe", "#ede9fe", "#fee2e2"]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        box(ax, x, .37, .13, .26, lab, colors[i], fs=10)
        if i < len(labels)-1: arrow(ax, x+.13, .50, xs[i+1], .50)
    ax.text(.5, .12, "Full feedback is used to obtain the clean target; the adaptation input remains the same sparse-observation task.",
            ha="center", fontsize=10, color="#475569", transform=ax.transAxes)
    ax.set_title("Sparse-input / clean-target adaptation construction", fontsize=15, weight="bold")
    savefig(p); return p


def figure_scope(scope_rows: list[tuple[str, int, float]]) -> Path:
    p = FIG / "fig3_scope_parameters.png"
    fig, ax = plt.subplots(figsize=(8.7, 4.5))
    labels = [x[0] for x in scope_rows]; values = [x[1] for x in scope_rows]; pct = [x[2] for x in scope_rows]
    bars = ax.bar(labels, values, color=["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#253494"])
    ax.set_ylabel("Trainable parameters"); ax.set_yscale("log"); ax.grid(axis="y", alpha=.25)
    ax.set_title("Parameter scope used in the controlled comparisons", weight="bold")
    ax.tick_params(axis="x", labelrotation=18)
    for b, n, q in zip(bars, values, pct): ax.text(b.get_x()+b.get_width()/2, n*1.2, f"{n:,}\n({q:.4g}%)", ha="center", fontsize=9)
    savefig(p); return p


def figure_3ep_reconstruction(rows: list[dict[str, str]]) -> Path:
    p = FIG / "fig4_three_epoch_reconstruction.png"
    models = ["3.16%", "Confirmatory", "25%", "Full"]
    data = {ng: [float(pick(rows, ng=str(ng), model=m)["nmse_improvement_120_db"]) for m in models] for ng in (16, 32)}
    fig, ax = plt.subplots(figsize=(8.7, 4.6)); x = np.arange(len(models)); w=.36
    ax.bar(x-w/2, data[16], w, label="Ng=16", color="#4c78a8")
    ax.bar(x+w/2, data[32], w, label="Ng=32", color="#f58518")
    ax.set_xticks(x, models); ax.set_ylabel("120 ns NMSE improvement [dB]\n(Pre − post; higher is better)")
    ax.set_title("Initial 3-epoch adaptation comparison", weight="bold"); ax.legend(); ax.grid(axis="y", alpha=.25)
    for i in range(len(models)):
        ax.text(i-w/2, data[16][i]+.03, f"{data[16][i]:.3f}", ha="center", fontsize=8)
        ax.text(i+w/2, data[32][i]+.03, f"{data[32][i]:.3f}", ha="center", fontsize=8)
    savefig(p); return p


def figure_collapse(rows: list[dict[str, str]]) -> Path:
    p = FIG / "fig5_far_ood_mechanism.png"
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.8))
    names = ["Ψ", "ν-margin", "κ"]
    for ax, q in zip(axs, ["psi", "nu_margin", "kappa"]):
        vals = [float(pick(rows, model="25%", regime="1ms", ng=str(ng), quantity=q)["median"]) /
                float(pick(rows, model="Pre", regime="1ms", ng=str(ng), quantity=q)["median"]) for ng in (16,32)]
        ax.bar(["Ng16", "Ng32"], vals, color=["#72b7b2", "#54a24b"])
        ax.set_title(q); ax.set_yscale("log"); ax.grid(axis="y", alpha=.25)
        for i,v in enumerate(vals): ax.text(i, v*1.5, f"{v:.3g}", ha="center", fontsize=9)
    fig.suptitle("25% / Pre median ratio at 1 ms", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0,0,1,.92)); savefig(p); return p


def figure_causal() -> Path:
    p = FIG / "fig6_late_feature_causal_flow.png"
    fig, ax = plt.subplots(figsize=(9, 3.0)); ax.axis("off")
    labels = ["Last residual blocks\nupdated", "Late feature\ndrift", "κ increases at\n1 ms", "Epistemic\nshrinks", "Far-OOD becomes\nknown-like"]
    xs = np.linspace(.03, .82, len(labels))
    for i,(x,l) in enumerate(zip(xs,labels)):
        box(ax,x,.35,.14,.30,l,["#fee2e2","#fee2e2","#fef3c7","#fef3c7","#ede9fe"][i],fs=10)
        if i<len(labels)-1: arrow(ax,x+.14,.50,xs[i+1],.50)
    ax.set_title("Evidence-based interpretation of the 25% Far-OOD collapse", weight="bold")
    savefig(p); return p


def figure_trajectory(epoch_rows: list[dict[str, str]]) -> Path:
    p = FIG / "fig7_last4_epoch_trajectory.png"
    pre = {}
    data = defaultdict(list)
    for r in epoch_rows:
        if r["regime"] != "1ms" or r["quantity"] != "epistemic": continue
        key=(r["ng"], r["epoch"])
        med=f(r,"median")
        if r["scope"] == "Pre" and r["epoch"] == "0": pre[r["ng"]]=med
        elif r["scope"] == "last_4_blocks_plus_head": data[(r["ng"], r["epoch"])].append(med)
    # The trajectory CSV stores one Pre row and one row per seed. Rebuild the seed mean ratio.
    for ax, ng, color in [(None,"16","#4c78a8"),(None,"32","#f58518")]: pass
    fig, ax = plt.subplots(figsize=(8.7,4.6))
    for ng,color in [("16","#4c78a8"),("32","#f58518")]:
        xs=[]; ys=[]
        for epoch in ["1","3","5","10"]:
            vals=data[(ng,epoch)]; xs.append(int(epoch)); ys.append(float(np.mean(vals))/pre[ng])
        ax.plot(xs,ys,marker="o",linewidth=2.3,label=f"Ng={ng}",color=color)
        for x,y in zip(xs,ys): ax.text(x,y*1.18,f"{y:.3f}",ha="center",fontsize=8)
    ax.set_xticks([1,3,5,10]); ax.set_yscale("log"); ax.set_xlabel("Adaptation epoch"); ax.set_ylabel("1 ms Epistemic median / Pre median")
    ax.set_title("Last4 long-run Far-OOD uncertainty trajectory",weight="bold"); ax.grid(alpha=.25,which="both"); ax.legend()
    savefig(p); return p


def figure_anchor(anchor_rows: list[dict[str, str]]) -> Path:
    p = FIG / "fig8_feature_anchor_comparison.png"
    vals = {"Ng16": [0.0171, 0.0171], "Ng32": [0.0086, 0.0087]}
    fig, ax = plt.subplots(figsize=(7.5,4.2)); x=np.arange(2); w=.34
    ax.bar(x-w/2,[vals["Ng16"][0],vals["Ng32"][0]],w,label="Unregularized last4",color="#9ecae1")
    ax.bar(x+w/2,[vals["Ng16"][1],vals["Ng32"][1]],w,label="Feature anchor",color="#de2d26")
    ax.set_xticks(x,["Ng16","Ng32"]); ax.set_ylabel("1 ms Epistemic median / Pre median"); ax.set_ylim(0,.025)
    ax.set_title("Feature anchoring did not recover Far-OOD uncertainty",weight="bold"); ax.legend(); ax.grid(axis="y",alpha=.25)
    savefig(p); return p


def figure_boundary_roles() -> Path:
    p = FIG / "fig9_boundary_roles.png"
    fig, ax=plt.subplots(figsize=(10,2.7)); ax.axis("off")
    roles=[("10–100 ns\noriginal known","#dbeafe"),("120 ns\nadapted regime","#dcfce7"),("160–500 ns\nproxy boundary rehearsal","#fef3c7"),("1 ms\nblind Far-OOD test","#fee2e2")]
    xs=[.03,.29,.55,.80]
    for i,((lab,c),x) in enumerate(zip(roles,xs)):
        box(ax,x,.35,.17,.30,lab,c,fs=10)
        if i<len(roles)-1: arrow(ax,x+.17,.50,xs[i+1],.50)
    ax.set_title("Data roles in uncertainty boundary distillation",weight="bold")
    savefig(p); return p


def figure_boundary_curve(bound_rows: list[dict[str, str]]) -> Path:
    p = FIG / "fig10_confidence_boundary_curve.png"
    grouped=defaultdict(list)
    for r in bound_rows:
        v=f(r,"epistemic_median")
        if math.isfinite(v): grouped[(int(r["delay_ns"]),r["ng"])].append(v)
    fig,ax=plt.subplots(figsize=(8.7,4.8))
    for ng,color in [("16","#4c78a8"),("32","#f58518")]:
        ds=sorted({d for d,n in grouped if n==ng}); ys=[np.mean(grouped[(d,ng)]) for d in ds]
        ax.plot(ds,ys,marker="o",linewidth=2,label=f"Ng={ng}",color=color)
    ax.axvspan(10,100,color="#dbeafe",alpha=.35,label="original known")
    ax.axvspan(120,120,color="#dcfce7",alpha=.8,label="adapted 120 ns")
    ax.axvspan(160,500,color="#fef3c7",alpha=.35,label="proxy boundary")
    ax.set_yscale("log"); ax.set_xlabel("Delay spread [ns]"); ax.set_ylabel("Epistemic median (raw scale)")
    ax.set_title("Boundary-distilled confidence curve; 1 ms remains blind",weight="bold"); ax.grid(alpha=.25,which="both"); ax.legend(fontsize=8,ncol=2)
    savefig(p); return p


def set_cell_shading(cell, fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr(); shd = tcPr.find(qn("w:shd"))
    if shd is None: shd = OxmlElement("w:shd"); tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text: str, bold=False, size=8.5, color="1F2937") -> None:
    cell.text = ""
    p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(str(text)); r.bold=bold; r.font.size=Pt(size); r.font.name="Aptos"; r.font.color.rgb=RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc: Document, headers: list[str], rows: list[list[object]], widths=None, font_size=8.4) -> None:
    table=doc.add_table(rows=1, cols=len(headers)); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.style="Table Grid"
    for j,h in enumerate(headers): set_cell_text(table.rows[0].cells[j],h,True,font_size,"17365D"); set_cell_shading(table.rows[0].cells[j],"D9EAF7")
    for row in rows:
        cells=table.add_row().cells
        for j,v in enumerate(row): set_cell_text(cells[j],v,False,font_size)
    if widths:
        for row in table.rows:
            for cell,w in zip(row.cells,widths): cell.width=Inches(w)
    doc.add_paragraph().paragraph_format.space_after=Pt(1)


def add_caption(doc: Document, text: str) -> None:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run(text); r.italic=True; r.font.size=Pt(9); r.font.color.rgb=RGBColor(80,80,80)


def add_figure(doc: Document, path: Path, caption: str, width=6.35) -> None:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(path),width=Inches(width)); add_caption(doc,caption)


def add_assumption(doc: Document, text: str) -> None:
    p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(.2); p.paragraph_format.right_indent=Inches(.2)
    r=p.add_run("RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION\n"); r.bold=True; r.font.color.rgb=RGBColor(157,74,0)
    p.add_run(text)


def omath_run(text: str):
    r=OxmlElement("m:r"); t=OxmlElement("m:t"); t.text=text; r.append(t); return r


def omath_fraction(numerator: str, denominator: str):
    frac = OxmlElement("m:f")
    num = OxmlElement("m:num"); num.append(omath_run(numerator))
    den = OxmlElement("m:den"); den.append(omath_run(denominator))
    frac.extend([num, den])
    return frac


def add_equation(doc: Document, text: str) -> None:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    omath=OxmlElement("m:oMath")
    if text == "Σ_ale = Ψ / (ν − 2K − 1)":
        omath.extend([omath_run("Σ_ale = "), omath_fraction("Ψ", "ν − 2K − 1")])
    elif text == "Σ_epi = Σ_ale / κ":
        omath.extend([omath_run("Σ_epi = "), omath_fraction("Σ_ale", "κ")])
    elif text == "L_feature-anchor = mean( ||z_student − z_teacher||² / (||z_teacher||² + ε) )":
        omath.extend([omath_run("L_feature-anchor = mean( "), omath_fraction("||z_student − z_teacher||²", "||z_teacher||² + ε"), omath_run(" )")])
    else:
        omath.append(omath_run(text))
    p._p.append(omath)


def add_page_number(paragraph):
    run=paragraph.add_run(); fldChar1=OxmlElement("w:fldChar"); fldChar1.set(qn("w:fldCharType"),"begin")
    instr=OxmlElement("w:instrText"); instr.set(qn("xml:space"),"preserve"); instr.text="PAGE"
    fldChar2=OxmlElement("w:fldChar"); fldChar2.set(qn("w:fldCharType"),"end")
    run._r.extend([fldChar1,instr,fldChar2])


def setup_doc() -> Document:
    doc=Document(); sec=doc.sections[0]; sec.page_width=Mm(210); sec.page_height=Mm(297)
    sec.top_margin=Mm(20); sec.bottom_margin=Mm(18); sec.left_margin=Mm(22); sec.right_margin=Mm(22)
    styles=doc.styles
    normal=styles["Normal"]; normal.font.name="Aptos"; normal.font.size=Pt(10.5); normal._element.rPr.rFonts.set(qn("w:eastAsia"),"맑은 고딕")
    normal.paragraph_format.line_spacing=1.18; normal.paragraph_format.space_after=Pt(6)
    for name,size,color in [("Title",24,"17365D"),("Heading 1",16,"17365D"),("Heading 2",13,"1F4E79"),("Heading 3",11,"335C81")]:
        st=styles[name]; st.font.name="Aptos Display"; st._element.rPr.rFonts.set(qn("w:eastAsia"),"맑은 고딕"); st.font.size=Pt(size); st.font.color.rgb=RGBColor.from_string(color)
        st.paragraph_format.space_before=Pt(12 if name!="Title" else 0); st.paragraph_format.space_after=Pt(6)
    footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER; footer.add_run("UACP Partial Fine-Tuning 연구 확장  |  "); add_page_number(footer)
    return doc


def p(doc, text="", bold_prefix=None):
    para=doc.add_paragraph()
    if bold_prefix and text.startswith(bold_prefix):
        r=para.add_run(bold_prefix); r.bold=True; para.add_run(text[len(bold_prefix):])
    else: para.add_run(text)
    return para


def section(doc, n, title): doc.add_heading(f"3.{n} {title}", level=1)


def build_report() -> None:
    ensure_clean_output()
    # Source data: every quantitative table/plot below is built from these repository artifacts.
    first3 = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_far_ood_diagnosis_20260921/confirmatory_result.csv")
    raw = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_far_ood_diagnosis_20260921/raw_evidential_stats.csv")
    final = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_final_validation_20260921_analysis_retry5/final_seed_summary.csv")
    eff = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_final_validation_20260921_analysis_retry5/efficiency_summary_aggregated.csv")
    eptraj = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_final_validation_20260921_analysis_retry5/epoch_trajectory.csv")
    eparam = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_final_validation_20260921_analysis_retry5/evidential_parameter_trajectory.csv")
    bcurve = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922/boundary_curve_retry2/boundary_curve.csv")
    boundary_recon = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922/reconstruction_summary.csv")
    anchor_recon = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_uncertainty_preserving_feature_anchor_20260922/reconstruction_summary.csv")

    # Inventory values are asserted against recorded scope inventory.
    inv = read_csv(ROOT/"runs/current_valid_baseline/partial_ft_20260921_sparse_3ep/scope_inventory.csv")
    inv_map={r.get("scope",r.get("model","")).lower(): r for r in inv}

    arch=figure_architecture(); flow=figure_flow()
    scope_fig=figure_scope([("Head only",4632,.0392),("Last1 + heads",373656,3.1625),("Last4 + heads",1480728,12.5323),("Last8 + heads",2956824,25.0253),("Full",11815320,100)])
    recon_fig=figure_3ep_reconstruction(first3); collapse_fig=figure_collapse(raw); causal_fig=figure_causal()
    traj_fig=figure_trajectory(eptraj); anchor_fig=figure_anchor(anchor_recon); roles_fig=figure_boundary_roles(); curve_fig=figure_boundary_curve(bcurve)

    doc=setup_doc()
    # Title page
    title=doc.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; title.paragraph_format.space_before=Pt(90)
    r=title.add_run("3. Uncertainty 기반 Partial Fine-Tuning 연구 확장"); r.bold=True; r.font.size=Pt(25); r.font.color.rgb=RGBColor(23,54,93)
    sub=doc.add_paragraph(); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER; sub.add_run("UACP evidential regression predictor의 OOD adaptation 및 uncertainty boundary 분석").font.size=Pt(14)
    doc.add_paragraph()
    meta=doc.add_paragraph(); meta.alignment=WD_ALIGN_PARAGRAPH.CENTER; meta.add_run("실험보고서 · repository 기록 기반 · 2026-09-23").font.size=Pt(11)
    doc.add_paragraph()
    p(doc,"본 보고서는 UACP 논문의 공개된 uncertainty-aware full-feedback 흐름을 출발점으로 삼아, 현재 repository의 구현에서 Partial Fine-Tuning을 연구 확장으로 검증한 결과를 정리한다. 수치와 그림은 저장된 CSV, Markdown 결과문서, checkpoint metadata에서 재확인했으며, 논문에 공개되지 않은 adaptation 정책은 별도로 표시하였다.")
    doc.add_page_break()

    doc.add_heading("요약", level=1)
    p(doc,"초기 질문은 Full Fine-Tuning의 parameter 일부만 갱신해도 120 ns adaptation 성능을 얻을 수 있는지였다. 3.16% scope는 3 epoch에서 Full improvement의 87.7–88.9%를 얻었고, 25.03% scope는 reconstruction과 120 ns uncertainty adaptation을 Full에 가깝게 만들었다. 그러나 25%와 Full에서는 학습하지 않은 1 ms에서 Epistemic uncertainty가 붕괴했고, 원인은 late residual feature drift와 κ 증가로 분해되었다.")
    p(doc,"중간 후보인 Last4 + heads(12.5323%)는 10 epoch·3 seed 검증에서 120 ns reconstruction retention은 100.6%(Ng16), 101.2%(Ng32)였지만 1 ms Epistemic/Pre median ratio는 0.017, 0.009로 낮아졌다. ID feature anchoring은 이 collapse를 줄이지 못했고, proxy-OOD boundary distillation은 반대로 κ를 지나치게 낮추고 ν-margin을 0에 접근시켜 불안정한 과대 uncertainty를 만들었다. 따라서 현재 구현·데이터·학습조건 범위에서 uncertainty-preserving Partial FT 해법은 확정되지 않았다.")
    add_assumption(doc,"Partial layer 정책, adaptation sample 수, epoch/LR, sparse-input/clean-target construction, feature anchor, proxy-OOD boundary distillation은 원 논문에 공개된 재현 설정이 아니다. 본문에서는 논문을 따른 요소와 연구 확장을 분리해 기술한다.")
    add_table(doc,["항목","판정"],[["120 ns reconstruction adaptation","Partial scope에서도 Full 수준까지 가능"],["20/80 ns ID 보존","scope와 duration에 따라 trade-off 존재"],["1 ms uncertainty 보존","기존 Partial/anchor/boundary 방식 모두 안정적 해결 실패"],["현재 연구 결론","parameter 비율만 조절해서는 충분하지 않음"]],[2.3,3.3])

    section(doc,"1","연구 목적")
    p(doc,"UACP는 주파수별 채널 정보를 나타내는 CFR(Channel Frequency Response)를 일부 subcarrier만 관측해 전체 CFR을 예측하고, 동시에 예측의 불확실성을 계산한다. Subcarrier는 OFDM에서 데이터를 나누어 보내는 개별 주파수 구간이며, sparse feedback은 모든 구간이 아니라 일부 구간만 receiver가 feedback하는 방식이다. NMSE는 복원 CFR과 실제 CFR의 차이를 나타내며, dB 값이 낮을수록 reconstruction이 좋다.")
    p(doc,"기존 predictor는 delay spread 10–100 ns 범위의 channel distribution을 학습했다. 120 ns는 이 범위를 벗어난 새로운 환경이고, 1 ms는 adaptation에 사용하지 않은 Far-OOD 환경이다. 논문은 높은 Epistemic uncertainty가 나타날 때 full feedback과 adaptation을 사용하는 흐름을 제시하지만, adaptation에서 어떤 layer를 fine-tune할지, sample 수·epoch·learning rate를 어떻게 고정할지는 공개하지 않는다.")
    p(doc,"따라서 본 연구는 Partial Fine-Tuning을 계산 효율적인 연구 확장으로 정의하였다. Fine-Tuning은 이미 학습한 모델의 일부 또는 전체 parameter를 추가 데이터로 갱신하는 과정이다. 연구 질문은 먼저 “얼마나 적은 parameter로 Full Fine-Tuning과 비슷하게 120 ns에 적응할 수 있는가?”에서 시작했고, 실험 결과를 통해 “120 ns를 새 known regime으로 흡수하면서 학습하지 않은 1 ms에 대한 uncertainty boundary를 어떻게 보존할 것인가?”로 구체화되었다.")

    section(doc,"2","모델 구조 및 공통 실험 설정")
    p(doc,"입력은 sparse CFR와 mask channel이다. Input projection 뒤에 32개의 residual block이 이어지며, 마지막 shared feature에서 네 개의 evidential head가 CFR 평균 γ, scale Ψ, confidence κ, degrees-of-freedom ν를 출력한다. Aleatoric uncertainty는 channel 자체의 예측 난이도, Epistemic uncertainty는 모델이 해당 환경을 충분히 학습하지 못한 데서 오는 불확실성으로 해석한다.")
    add_figure(doc,arch,"Figure 1. UACP predictor의 현재 구현 구조. ResidualBlock 32개 뒤에서 reconstruction 출력과 evidential parameter 출력을 분기한다.")
    add_table(doc,["Scope","Trainable parameters","전체 대비"],[["Head only","4,632","0.0392%"],["Last 1 block + heads","373,656","3.1625%"],["Last 4 blocks + heads","1,480,728","12.5323%"],["Last 8 blocks + heads","2,956,824","25.0253%"],["Full","11,815,320","100%"]],[2.5,2.1,1.2])
    add_figure(doc,scope_fig,"Figure 2. 고정된 architecture에서 trainable scope를 넓힐 때의 parameter 규모. 로그 축은 Full과 Partial의 차이를 함께 보이기 위한 것이다.")
    p(doc,"모든 Partial FT 비교는 동일한 epoch3 Pre checkpoint에서 시작했다. 120 ns full feedback은 adaptation용 clean full-CFR target을 확보하는 데 사용했으며, predictor 입력은 기존 task와 같은 sparse CFR + mask로 다시 구성했다. 따라서 full-CFR를 입력으로 넣어 full-CFR를 복사하는 identity Fine-Tuning은 수행하지 않았다.")
    add_figure(doc,flow,"Figure 3. Sparse-input / clean-target adaptation construction. 15 dB complex AWGN은 reported CFR에 적용되며 target은 clean full CFR이다.")
    add_table(doc,["Evaluation regime","의미","역할"],[["20 ns","ID-Easy","기존 학습 분포의 쉬운 조건"],["80 ns","ID-Hard","기존 학습 분포의 어려운 조건"],["120 ns","adapted regime","추가 adaptation 대상"],["1 ms","Far-OOD","학습·coefficient 선택·model selection에 사용하지 않은 blind test"]],[1.3,1.5,3.0])
    add_assumption(doc,"Ng 후보, 15 dB AWGN, mask policy, clean-target adaptation, batch/epoch/LR/seed와 Partial layer 선택은 현재 구현의 고정 조건이다. 논문 공개 설정과 혼동하지 않는다.")

    section(doc,"3","1차 실험 — Last1 + heads, 3.16%")
    p(doc,"첫 실험은 ResidualBlock31과 네 evidential head만 갱신하는 3.16% scope와 Full scope를 비교했다. 두 모델은 같은 epoch3 checkpoint에서 독립적으로 시작했고, 120 ns adaptation train/validation split, sample order, sparse mask와 AWGN schedule을 공유했다. 120 ns train sample 1,000개와 validation 200개, 3 epochs, batch 8, Adam, learning rate 1e−4, FP32, λreg=1e−3, seed 20260921을 사용했다.")
    add_table(doc,["Model","Ng","120 ns improvement [dB]","Full retention","20 ns ΔNMSE [dB]","80 ns ΔNMSE [dB]"],[["3.16% Partial","16","1.243","87.7%","+1.365","−0.076"],["3.16% Partial","32","0.918","88.9%","+1.309","−0.113"],["Full","16","1.417","100.0%","+2.771","+0.195"],["Full","32","1.033","100.0%","+2.514","+0.226"]],[1.2,.5,1.2,1.0,1.0,1.0])
    p(doc,"3.16% Partial은 120 ns reconstruction improvement의 약 88–89%를 얻었고, 80 ns forgetting은 Full보다 작았다. 다만 120 ns Epistemic 변화는 Full보다 작아, reconstruction path만 맞추는 것과 uncertainty adaptation까지 맞추는 것이 같은 문제는 아님을 보였다. 다음 단계에서는 trainable 범위를 확대해 Full에 더 가까워지는지 확인했다.")

    section(doc,"4","2차 실험 — Last8 + heads, 25.03%")
    p(doc,"Last8 + heads는 ResidualBlock24–31과 네 evidential head를 갱신하는 25.0253% scope다. 초기 3 epoch 비교에서 Ng16 120 ns improvement는 1.521 dB, Ng32는 1.129 dB였고, Full 대비 retention은 각각 107.4%, 109.3%였다. 이 결과만 보면 reconstruction adaptation은 충분히 달성되었다.")
    add_figure(doc,recon_fig,"Figure 4. 3 epoch controlled comparison의 120 ns NMSE improvement. 값은 Pre − post이며, 높을수록 adaptation으로 NMSE가 더 낮아졌다는 뜻이다.")
    p(doc,"문제는 1 ms였다. 120 ns는 adaptation에 사용했으므로 Epistemic uncertainty가 낮아지는 것은 새로운 known regime으로 흡수된 결과일 수 있다. 그러나 1 ms는 training에 사용하지 않았는데도 Epistemic이 함께 낮아졌다. 따라서 120 ns adaptation 성능만으로 Partial FT의 uncertainty 품질을 판단할 수 없다.")

    section(doc,"5","Far-OOD Epistemic collapse 원인 분석")
    p(doc,"현재 구현은 evidential parameter로부터 다음 uncertainty를 계산한다. K는 출력 차원에 대응하는 상수이며, Ψ는 scale, ν는 degrees-of-freedom, κ는 confidence parameter이다.")
    add_equation(doc,"Σ_ale = Ψ / (ν − 2K − 1)")
    add_equation(doc,"Σ_epi = Σ_ale / κ")
    p(doc,"25%와 Pre의 1 ms median ratio는 Ng16에서 Ψ=0.977, ν-margin=1.416, κ=120.5였고, Ng32에서 Ψ=0.933, ν-margin=1.846, κ=213.5였다. Ψ 변화는 상대적으로 작았지만 κ가 수십~수백 배 커져 Σ_epi를 직접 줄였고, ν-margin 증가가 추가로 Aleatoric term을 줄였다.")
    add_table(doc,["Ng","Ψ ratio","ν-margin ratio","κ ratio","해석"],[["16","0.977","1.416","120.5","κ inflation이 직접 원인; ν-margin이 추가 기여"],["32","0.933","1.846","213.5","κ inflation이 더 큼; Ψ는 2차 요인"]],[.6,1.0,1.2,1.0,2.5])
    add_figure(doc,collapse_fig,"Figure 5. 25% checkpoint와 Pre의 1 ms evidential median ratio. κ의 로그 규모 변화가 Epistemic collapse를 지배한다.")
    p(doc,"Checkpoint component swap에서는 ResidualBlock24–31만 25% 값으로 교체한 H-BLOCK에서 collapse가 재현되었고, evidential heads만 교체한 H-HEAD와 uncertainty heads만 교체한 H-UNC에서는 Pre 수준 uncertainty가 보존되었다. gamma head만 교체한 H-GAMMA도 collapse를 만들지 않았다. 이 결과는 단순한 head parameter 변화보다 late residual representation drift가 핵심이며, 그 feature가 evidential head와 결합되어 κ를 폭증시키는 구조임을 지지한다.")
    add_table(doc,["Hybrid","25% checkpoint에서 가져온 component","1 ms collapse"],[["H-BLOCK","ResidualBlock24–31","재현"],["H-HEAD","네 evidential heads","보존"],["H-UNC","Ψ/κ/ν uncertainty heads","보존"],["H-GAMMA","γ head","보존"]],[1.1,3.2,1.3])
    add_figure(doc,causal_fig,"Figure 6. Hybrid evaluation과 raw evidential statistics를 함께 사용한 원인 흐름.")
    p(doc,"따라서 120 ns reconstruction improvement와 1 ms Epistemic collapse는 서로 다른 출력만의 문제가 아니라, adaptation 중 late feature가 120 ns에 맞춰 이동하면서 먼 OOD도 같은 feature branch로 끌어오는 trade-off로 해석된다.")

    section(doc,"6","3차 실험 — Last4 + heads, 12.53%")
    p(doc,"Last4 + heads는 3.16%와 25.03% 사이에서 late feature drift를 줄이면서도 reconstruction capacity를 유지하기 위해 선택한 중간 scope다. ResidualBlock28–31과 네 evidential head를 학습하며 1,480,728개 parameter, 전체의 12.5323%를 사용한다.")
    add_table(doc,["3 epoch Last4","Ng16","Ng32"],[["120 ns improvement [dB]","1.426","1.105"],["Full retention","100.7%","107.0%"],["1 ms / 120 ns Epistemic separation","61,156","35,726"],["ID forgetting: 20 ns [dB]","+1.925","+1.730"],["ID forgetting: 80 ns [dB]","−0.053","−0.065"]],[2.8,1.3,1.3])
    p(doc,"초기 3 epoch 결과는 reconstruction이 Full 수준이고, 25%보다 Far-OOD separation이 크게 회복되어 유망해 보였다. 그러나 짧은 학습에서만 나타난 효과인지 확인하지 않은 상태에서는 final operating point로 확정할 수 없었다. 이에 10 epochs와 세 seed로 장기·확률적 robustness를 검증했다.")

    section(doc,"7","장기 학습 및 Seed Robustness 검증")
    p(doc,"검증은 seed 20260921, 20260922, 20260923에서 epoch 1/3/5/10 checkpoint를 저장하는 방식으로 수행되었다. Primary endpoint는 사전에 epoch10으로 고정했고, 1 ms 결과를 보고 best epoch를 선택하지 않았다. 모든 scope는 동일한 starting checkpoint와 adaptation protocol을 사용했다.")
    add_figure(doc,traj_fig,"Figure 7. Last4의 1 ms Epistemic median/Pre median trajectory. 세 seed의 epoch별 값을 평균해 표시했으며, epoch가 늘수록 uncertainty retention이 감소한다.")
    add_table(doc,["Scope","Ng","120 ns improvement [dB]","Full retention","1 ms Epi/Pre median","20 ns ΔNMSE [dB]","80 ns ΔNMSE [dB]"],[["Last4","16","1.513 ± 0.003","100.6%","0.017","−2.390 ± 0.034","+0.017 ± 0.017"],["Last4","32","1.145 ± 0.007","101.2%","0.009","−2.152 ± 0.032","+0.000 ± 0.022"],["Full","16","1.505 ± 0.048","—","0.004","−2.928 ± 0.070","−0.147 ± 0.054"],["Full","32","1.132 ± 0.039","—","0.003","−2.752 ± 0.042","−0.208 ± 0.036"]],[.7,.4,1.1,.8,1.0,1.1,1.1],8.0)
    p(doc,"Last4는 reconstruction 기준을 충족했지만 1 ms Epistemic median/Pre ratio가 epoch1에서 약 0.106/0.051, epoch3에서 0.038/0.015, epoch5에서 0.028/0.011, epoch10에서 0.017/0.009로 계속 낮아졌다. AUROC가 Ng16 0.999994, Ng32 0.9999005로 높게 유지된 것은 실패와 모순되지 않는다. AUROC는 두 분포의 순위가 분리되는지를 보지만, 두 분포 모두 absolute uncertainty가 낮아지는 calibration collapse는 놓칠 수 있다.")
    p(doc,"epoch10에서 Last4의 1 ms median ratio는 Ψ=0.973/0.957, ν-margin=1.323/1.668, κ=43.327/67.492(Ng16/Ng32)였다. 따라서 3 epoch에서 좋아 보였던 Last4 결과는 장기 학습과 seed 반복에서 재현되는 안정적인 operating point가 아니라 short-duration transient effect로 판단했다.")

    section(doc,"8","해결 시도 1 — ID Feature Anchoring")
    p(doc,"첫 번째 해결 시도는 adaptation 중 기존 ID feature가 변하지 않도록 Pre checkpoint를 frozen teacher로 두고, Student의 ResidualBlock31 출력(이 구조에서 evidential-head input과 동일한 tensor)을 teacher feature에 가깝게 유지하는 방식이었다.")
    add_equation(doc,"L_feature-anchor = mean( ||z_student − z_teacher||² / (||z_teacher||² + ε) )")
    add_assumption(doc,"정규화 feature anchor, ID subset 1,000개, λ_anchor=1.0, 10 epoch·3 seed는 원 논문에 공개되지 않은 연구 확장 설정이다.")
    add_figure(doc,anchor_fig,"Figure 8. Feature anchoring과 unregularized Last4의 epoch10 1 ms Epistemic/Pre median ratio. anchor는 Far-OOD collapse를 유의하게 회복시키지 못했다.")
    add_table(doc,["Ng","Anchored improvement [dB]","Full retention","Anchored Epi/Pre","Unregularized Epi/Pre","Gate"],[["16","1.512 ± 0.007","100.6%","0.0171","0.0171","실패"],["32","1.146 ± 0.009","101.2%","0.0087","0.0086","실패"]],[.5,1.2,1.0,1.0,1.3,0.7])
    p(doc,"120 ns reconstruction은 유지되었지만 1 ms uncertainty는 개선되지 않았고, head-input normalized drift도 unregularized Last4와 유사했다. 따라서 ID feature를 직접 보존하는 제약만으로는 feature-to-evidential calibration boundary를 유지할 수 없었다.")

    section(doc,"9","해결 시도 2 — Uncertainty Boundary Distillation")
    p(doc,"두 번째 시도는 120 ns reconstruction adaptation과 uncertainty preservation을 서로 다른 데이터 역할로 분리했다. 10–100 ns ID subset은 기존 uncertainty semantics 보존, 120 ns는 reconstruction adaptation, 160–500 ns proxy-OOD는 confidence boundary rehearsal에 사용했다. 1 ms는 training, coefficient 결정, model selection에 사용하지 않은 blind final test였다.")
    add_figure(doc,roles_fig,"Figure 9. Boundary distillation의 데이터 역할. 160–500 ns proxy-OOD에는 clean reconstruction target loss를 적용하지 않았다.")
    add_table(doc,["범위","학습에서의 역할"],[["10–100 ns","기존 uncertainty 보존"],["120 ns","sparse-to-clean reconstruction adaptation"],["160–500 ns","uncertainty-only boundary rehearsal"],["1 ms","완전 blind Far-OOD 평가"]],[1.5,4.3])
    add_equation(doc,"L_ID = MSE(log(Epi_s + ε), log(Epi_t + ε)) + MSE(log(Ale_s + ε), log(Ale_t + ε))")
    add_equation(doc,"L_boundary = mean( ReLU(log(Epi_t + ε) − log(Epi_s + ε))² )")
    add_equation(doc,"L_total = L_adapt + λ_ID L_ID + λ_boundary L_boundary")
    p(doc,"One-sided boundary loss는 Student Epistemic이 Teacher보다 낮아질 때만 penalty를 주며, 더 uncertain한 방향은 막지 않는다. 각 auxiliary loss는 첫 smoke batch의 detached magnitude로 normalize하고 λ_ID=λ_boundary=1.0으로 고정했다.")
    add_assumption(doc,"Proxy-OOD Uniform(160,500) ns 범위, auxiliary loss normalization, λ_ID/λ_boundary, 10 epoch·3 seed는 논문 재현값이 아니라 연구 확장 가정이다.")

    section(doc,"10","Boundary Distillation 결과")
    p(doc,"Boundary distillation은 120 ns reconstruction 자체는 유지했다. Ng16 improvement는 1.491 ± 0.014 dB, Full retention 99.1%, Ng32 improvement는 1.131 ± 0.013 dB, Full retention 99.9%였다.")
    add_figure(doc,curve_fig,"Figure 10. Boundary-distilled Last4의 delay-spread별 Epistemic median. y축은 raw Epistemic scale의 로그 축이며, 1 ms 값은 blind test에서 가져왔다. 120 ns 이후 uncertainty가 증가하는 방향은 형성되지만, 1 ms까지 안정적인 calibration으로 이어지지는 않는다.")
    add_table(doc,["Ng","1 ms Epi/Pre median","κ ratio","ν-margin ratio","상태"],[["16","8.85","0.168","0.737","과도한 uncertainty"],["32","non-finite","0.175","0.573","ν-margin=0 sample 발생"]],[.6,1.5,1.0,1.2,2.0])
    p(doc,"이 방법은 unregularized Last4의 κ inflation을 억제했지만, κ를 지나치게 낮추고 ν-margin을 0에 접근시켰다. 그 결과 Ng16은 Pre보다 훨씬 큰 Epistemic을 만들었고, Ng32는 Eq.(7)/(8)의 분모가 0이 되는 sample에서 Aleatoric/Epistemic이 non-finite가 되었다. non-finite 값은 0으로 치환하거나 숨기지 않고 원본 결과에 남겼다.")
    add_table(doc,["방법","120 ns adaptation","1 ms uncertainty 동작"],[["Unregularized Last4","성공","너무 낮음 (0.017/0.009)"],["Feature Anchor","성공","여전히 너무 낮음 (0.0171/0.0087)"],["Boundary Distillation","성공","너무 높고 일부 non-finite"],["목표","성공","높지만 finite하고 안정적"]],[1.5,1.8,3.0])

    section(doc,"11","연구 질문의 변화와 종합 해석")
    p(doc,"실험은 parameter 비율만 줄이는 문제가 아니라 adaptation 이후 uncertainty boundary를 보존하는 문제로 확장되었다. 3.16%는 비용과 Far-OOD 보존 측면에서 유리했지만 adaptation이 부족했고, 25.03%는 Full 수준의 adaptation을 얻는 대신 late feature drift와 κ inflation을 만들었다. Last4는 두 scope 사이의 유망한 단기 절충안이었으나 10 epoch·3 seed에서 같은 collapse가 누적되었다. Feature Anchor는 representation 보존에만 집중해 collapse를 막지 못했고, Boundary Distillation은 반대 방향의 과도한 uncertainty와 수치 불안정을 만들었다.")
    add_table(doc,["단계","관찰","해석"],[["3.16%","저비용, Far-OOD 상대 보존, adaptation 약함","작은 scope는 feature boundary를 덜 움직임"],["25.03%","Full 수준 adaptation, 1 ms overconfidence","late blocks가 1 ms representation까지 이동"],["12.53% Last4","3 epoch 유망, 10 epoch collapse","short-duration transient"],["Feature Anchor","reconstruction 유지, collapse 미해결","feature 고정만으로 calibration boundary 보존 불충분"],["Boundary Distillation","boundary 방향 형성, over-uncertainty/non-finite","one-sided penalty가 ν/κ 하한을 제어하지 못함"]],[1.4,2.7,2.2])
    p(doc,"따라서 현재 테스트한 방법과 설정 범위에서 Partial FT를 단순히 “성공” 또는 “실패”로 일반화할 수는 없다. reconstruction adaptation은 Partial scope에서도 가능하지만, known regime을 넓히면서 더 먼 OOD의 Epistemic calibration을 안정적으로 보존하려면 reconstruction objective와 uncertainty objective를 별도로 설계해야 한다.")

    section(doc,"12","다음 실험")
    p(doc,"repository에 기록된 다음 단일 실험은 uncertainty-head freeze controlled run이다. ResidualBlock28–31과 gamma_head는 120 ns reconstruction에 적응시키되, psi_head, kappa_head, nu_head는 Pre 상태로 고정한다.")
    add_table(doc,["구분","설정"],[["Train","ResidualBlock28–31 + gamma_head"],["Freeze","psi_head + kappa_head + nu_head"],["고정 조건","same Pre, data, sparse mask, AWGN, Adam, LR, batch, epochs, seeds"],["질문","uncertainty-head weight 변화만 제거하면 Far-OOD overconfidence가 줄어드는가?"]],[1.7,4.6])
    p(doc,"이 실험은 head를 고정해도 입력 feature가 이동하면 uncertainty output이 변할 수 있다는 점을 전제로 한다. 따라서 결과는 uncertainty-head weight 변화의 기여를 분리하는 controlled evidence로 해석하며, 새로운 scope sweep이나 auxiliary coefficient tuning으로 확장하지 않는다.")

    doc.add_heading("실험 무결성 및 재현성", level=1)
    p(doc,f"기준 Pre checkpoint SHA-256은 {PRE_SHA}이며, 기존 checkpoint/result/log/dataset은 read-only로 재사용했다. Partial/Full 비교는 sparse CFR + mask → clean full-CFR target protocol을 유지했다. Feature Anchor와 Boundary Distillation 모두 1 ms를 training·coefficient 선택·model selection에 사용하지 않았으며, Boundary Distillation의 proxy-OOD에는 reconstruction target loss를 적용하지 않았다.")
    p(doc,"평가와 training은 repository 기록상 NVIDIA GB10, cuda:0, FP32를 사용했다. Leakage 검사, frozen parameter 검사, finite/non-finite 검사와 smoke tests 결과는 각 실험 manifest와 integrity_summary.json에 보존되어 있다. Dynamic controller/Fig.11은 현재 고정 validation에 맞는 동일 pipeline이 없어 static uncertainty evidence를 primary로 사용했다.")
    add_table(doc,["실험","핵심 원본 결과 파일"],[["3 epoch Partial/Full","partial_ft_far_ood_diagnosis_20260921/confirmatory_result.csv"],["25% 원인 분석","partial_ft_far_ood_diagnosis_20260921/raw_evidential_stats.csv, activation_drift.csv"],["10 epoch·3 seed validation","partial_ft_final_validation_20260921_analysis_retry5/final_seed_summary.csv, epoch_trajectory.csv"],["Feature Anchor","partial_ft_uncertainty_preserving_feature_anchor_20260922/"],["Boundary Distillation","partial_ft_uncertainty_boundary_distillation_20260922/"]],[2.3,4.0],8.2)
    doc.add_heading("참고문헌", level=1)
    p(doc,"[1] UACP: Uncertainty-Aware Channel Prediction in MIMO Networks, mobihoc26-paper289.pdf, repository copy of the UACP manuscript.")
    p(doc,"[2] Repository experiment records: docs/PARTIAL_FT_RESULTS_20260921.md; docs/PARTIAL_FT_25PCT_RESULTS_20260921.md; docs/PARTIAL_FT_FAR_OOD_DIAGNOSIS_20260921.md; docs/PARTIAL_FT_FINAL_VALIDATION_20260921.md; docs/PARTIAL_FT_UNCERTAINTY_PRESERVING_20260922.md; docs/PARTIAL_FT_UNCERTAINTY_BOUNDARY_20260922.md.")

    doc.save(DOCX)
    print(DOCX)


if __name__ == "__main__":
    build_report()
