"""Accessible static scouting cards for the four weekly prop lists."""

from __future__ import annotations

import html

e = html.escape

TITLES = {
    "role": "Role and opportunity",
    "offensive_intent": "Offensive intention and tempo",
    "formation": "Formations",
    "personnel": "Offensive personnel",
    "concepts": "Motion, play action, RPO and screens",
    "opponent_coverage": "Opponent coverage and shells",
    "opponent_front_and_pressure": "Opponent fronts, packages and pressure",
    "offense_response": "Offensive answers to the looks",
    "opponent_response": "Opponent efficiency in those looks",
    "individual_look_response": "Individual coverage, pressure, fronts and tracking",
    "run_game_and_trenches": "Run game, run points and line play",
    "drive_finishing": "Scoring-range access and drive finishing",
    "availability": "Injuries and replacement roles",
    "regime": "Staff continuity and recent changes",
    "game_environment": "Scoring, stadium, rest and travel",
    "unobserved_assignments": "Charting still needed",
}


def _label(key: str) -> str:
    return (
        key.replace("_", " ")
        .replace("epa", "EPA")
        .replace("rpo", "RPO")
        .replace("qb", "QB")
        .replace("td", "TD")
        .replace("fg", "FG")
        .replace("pat", "PAT")
    )


def _value(key: str, value) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if any(part in key for part in ("rate", "share", "frequency", "covered_mass")):
            return f"{value:.1%}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _leaves(value, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"used_to_change_projection", "mechanism", "family", "status"}:
                continue
            yield from _leaves(child, f"{prefix} · {_label(key)}" if prefix else _label(key))
    elif isinstance(value, list):
        if not value:
            yield prefix, "Unavailable"
        elif all(not isinstance(x, (list, dict)) for x in value):
            yield prefix, ", ".join(str(x) for x in value)
        else:
            for i, child in enumerate(value):
                name = (
                    (
                        child.get("player_name")
                        or child.get("look")
                        or child.get("coverage")
                        or child.get("name")
                    )
                    if isinstance(child, dict)
                    else None
                )
                yield from _leaves(child, f"{prefix} · {name or i + 1}")
    else:
        yield prefix, _value(prefix.replace(" ", "_"), value)


def _evidence(section: dict) -> str:
    if "values" in section:
        rows = []
        for key, value in section["values"].items():
            source = section["provenance"].get(key, {})
            years = ", ".join(str(y) for y in source.get("source_seasons", [])) or "undated"
            participation = source.get("participation_source_seasons", [])
            sample = source.get("plays")
            note = f"{source.get('source', '')}; seasons {years}"
            if participation:
                note += "; coverage/personnel " + ", ".join(str(y) for y in participation)
            if sample is not None:
                note += f"; {sample} unit plays (split denominator may differ)"
            rows.append(
                f'<tr><th scope="row">{e(_label(key))}</th>'
                f"<td>{e(_value(key, value))}</td><td>{e(note)}</td></tr>"
            )
        if section.get("missing_fields"):
            rows.append(
                '<tr><th scope="row">Unavailable fields</th><td colspan="2">'
                + e(", ".join(_label(k) for k in section["missing_fields"]))
                + "</td></tr>"
            )
        table = "<table><thead><tr><th>Measure</th><th>Value</th><th>Evidence date / sample</th>"
        table += "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    else:
        rows = [
            f'<tr><th scope="row">{e(key)}</th><td>{e(value)}</td></tr>'
            for key, value in _leaves(section)
        ]
        table = "<table><tbody>" + "".join(rows) + "</tbody></table>"
    title = TITLES.get(section["family"], _label(section["family"]))
    return (
        f'<details class="wp-evidence"><summary>{e(title)} '
        f"<span>{e(section['status'])}</span></summary>"
        f'<p>{e(section.get("mechanism", ""))}</p><div class="tablewrap">{table}</div></details>'
    )


def _items(values: list[str], empty: str) -> str:
    return "<ul>" + "".join(f"<li>{e(value)}</li>" for value in (values or [empty])) + "</ul>"


def _card(row: dict) -> str:
    threshold = row["threshold"]
    direction = row["selection"]
    research = row["threshold_source"] == "research_milestone"
    if research:
        hurdle = int(threshold + 0.5)
        call = f"{hurdle}+ {row['market'].lower()}"
        source = "Research milestone · no posted book line"
    else:
        call = f"{direction} {threshold:g} {row['market'].lower()}"
        source = row["book"] + (" · prices unavailable" if not row["priced"] else "")
    interval = row.get("hit_probability_interval")
    probability = (
        f"{interval[0]:.0%}–{interval[1]:.0%} bound"
        if interval
        else f"{row['hit_probability']:.1%}"
    )
    basis = {
        "line_calibrated": "Line-calibrated model",
        "raw_distribution": "Uncalibrated milestone model",
        "poisson_assumption": "Unvalidated Poisson model",
        "dependence_bound": "Dependence bound",
    }[row["probability_basis"]]
    scenarios = "".join(
        f'<tr><th scope="row">{e(s["scenario"])}</th><td>×{s["mean_scale"]:.2f}</td>'
        f"<td>{s['hit_probability']:.1%}</td><td>{s['push_probability']:.1%}</td></tr>"
        for s in row["scenarios"]
    )
    details = "".join(_evidence(section) for section in row["evidence"])
    return f"""<article class="wp-card" data-position="{e(row["position"])}">
<div class="wp-card-head"><span class="rank">{row["rank"]:02d}</span>
<div><b>{e(row["player"])}</b><span class="dim"> {e(row["team"])} vs {e(row["opponent"])}</span>
<p class="wp-call">{e(call)}</p></div>
<div class="wp-likelihood"><strong>{row["conservative_hit_probability"]:.1%}</strong>
<span>Stress likelihood</span></div></div>
<div class="wp-meta">{e(source)} · Mean {row["model_mean"]:.1f} · Base {e(probability)} ·
{e(basis)} · {e(row["schematic_assessment"])}</div>
<details class="wp-analysis"><summary>Scouting case, threshold and failure paths</summary>
<p>{e(row["thesis"])}</p>
<div class="wp-two"><div><h4>Supports the threshold</h4>
{_items(row["support"], "Directional scheme support unavailable; role-based research only.")}</div>
<div><h4>Points the other way</h4>
{_items(row["conflicts"], "No opposing measured scheme delta in available evidence.")}</div></div>
<h4>What breaks the case</h4>{_items(row["failure_paths"], "Recheck the role and line.")}
<h4>Opportunity stress test</h4>
<p>{e("; ".join(row["stress_reasons"]))}. These are sensitivity assumptions.</p>
<div class="tablewrap"><table><thead><tr><th>Scenario</th><th>Mean scale</th>
<th>Hit likelihood</th><th>Push</th></tr></thead><tbody>{scenarios}</tbody></table></div>
<p class="dim">{e(row["probability_assumption"])}. Quote timestamp:
{e(row.get("quote_updated_at") or "Unavailable")}. {e(row["action"])}.</p>
<h4>Complete evidence dossier</h4>{details}</details></article>"""


def render(report: dict) -> str:
    groups = []
    for position, group in report["groups"].items():
        cards = "".join(_card(row) for row in group["rows"])
        note = (
            f"{group['quoted']} posted lines · {group['research_thresholds']} research milestones"
            f" · {group['candidate_players']} eligible players"
        )
        if group["shortfall"]:
            note += f" · {group['shortfall']} unavailable slots"
        if not cards:
            cards = f'<p class="dim">{e(group["availability_note"])}</p>'
        groups.append(
            f'<details class="wp-group" id="weekly-props-{position.lower()}" open>'
            f"<summary>Top 10 {e(group['label'])} props "
            f"<span>{group['published']}/10 · {e(note)}</span></summary>{cards}</details>"
        )
    context = report.get("context_status") or {}
    context_note = f"MLBMA scouting: {context.get('state', 'unavailable')}"
    if context.get("generated_at_utc"):
        context_note += f" · {context['generated_at_utc']}"
    return f"""<section id="weekly-props">
<div class="sec-head"><span class="kicker">Weekly player scouting</span>
<h2>Week {report["week"]} · RB, QB, WR &amp; kicking top tens</h2>
<p class="blurb">Ten distinct players per position when eligible evidence is available.
Posted line tiers come first; each tier ranks by the lowest model hit likelihood across
disclosed workload and efficiency scenarios. Expand a player for the offensive plan,
opponent responses, individual splits, threshold requirements and failure paths.</p>
<p class="fine">{e(report["stress_note"])}</p><p class="dim">{e(context_note)} · RESEARCH_ONLY</p>
<nav class="wp-jump" aria-label="Weekly prop positions">
<a href="#weekly-props-rb">RB</a><a href="#weekly-props-qb">QB</a>
<a href="#weekly-props-wr">WR</a><a href="#weekly-props-k">Kicking</a>
<a href="weekly-props.json">Evidence JSON</a></nav></div>{"".join(groups)}</section>"""


CSS = """
.wp-group{border:1px solid var(--border-soft);border-radius:10px;margin:18px 0;
padding:0 16px;background:var(--ca-panel-glass)}
.wp-group>summary{padding:18px 0;font-weight:800;cursor:pointer;font-size:1.1rem}
.wp-group>summary span{display:block;color:var(--text-3);font-size:.78rem;font-weight:400;
margin:7px 0 0 18px;line-height:1.5}
.wp-card{border-top:1px solid var(--border-soft);padding:16px 0}
.wp-card-head{display:grid;grid-template-columns:34px minmax(0,1fr) auto;gap:12px;
align-items:start}.wp-card-head b{font-size:1rem}.wp-call{margin:4px 0 0;font-weight:700;
color:var(--v-light)}.wp-likelihood{text-align:right}.wp-likelihood strong{display:block;
font-size:1.2rem;color:var(--v-light)}.wp-likelihood span{font-size:.68rem;color:var(--text-3)}
.wp-meta{font-size:.78rem;color:var(--text-3);margin:10px 0;line-height:1.6}
.wp-analysis>summary,.wp-evidence>summary{cursor:pointer;min-height:44px;display:list-item;
padding:12px 0;color:var(--text-2);font-size:.82rem}
.wp-analysis p,.wp-analysis li{font-size:.84rem;color:var(--text-2);line-height:1.65}
.wp-analysis h4{margin:18px 0 8px}.wp-analysis ul{padding-left:20px}
.wp-two{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.wp-evidence{border-top:1px solid var(--border-soft)}
.wp-evidence summary span{color:var(--text-3);font-size:.72rem;margin-left:8px}
.wp-analysis table{width:100%;border-collapse:collapse;font-size:.78rem}
.wp-analysis td,.wp-analysis th{padding:8px;text-align:left;vertical-align:top;
white-space:normal;border-bottom:1px solid var(--border-soft);overflow-wrap:anywhere}
.wp-analysis th{color:var(--text-3);font-weight:500;max-width:320px}
.wp-jump{display:flex;gap:8px;flex-wrap:wrap}.wp-jump a{min-height:44px;display:flex;
align-items:center;padding:0 14px;border:1px solid var(--border-soft);border-radius:7px;
color:var(--text-2);text-decoration:none}
@media(max-width:600px){.wp-group{padding:0 12px}.wp-card-head{gap:8px;
grid-template-columns:24px minmax(0,1fr)}.wp-likelihood{grid-column:2;text-align:left;
display:flex;gap:10px;align-items:center}.wp-two{grid-template-columns:1fr;gap:0}
.wp-group>summary span{margin-left:0}.wp-card-head .dim{display:block}}
"""
