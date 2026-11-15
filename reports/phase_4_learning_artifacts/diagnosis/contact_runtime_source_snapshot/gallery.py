"""Offline floor-plan gallery and explicit human review contract."""
import hashlib
import html
import json
import math
from pathlib import Path

from .contracts import ContractError, validate_scene


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _keys(value, keys, path):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ContractError(f"{path}: expected exactly {', '.join(sorted(keys))}")


def validate_review(review, manifest):
    """Reject incomplete/mismatched evidence; return an explicit human gate result."""
    _keys(review, {"schema_version", "gallery_id", "reviewer", "reviewed_at", "source_scene_sha256", "decisions"}, "review")
    if review["schema_version"] != "ares.floor_plan_review.v1":
        raise ContractError("review.schema_version: unsupported version")
    if review["gallery_id"] != manifest["gallery_id"]:
        raise ContractError("review.gallery_id: wrong gallery")
    if review["source_scene_sha256"] != manifest["source_scene_sha256"]:
        raise ContractError("review.source_scene_sha256: scene digests mismatch")
    for name in ("reviewer", "reviewed_at"):
        if not isinstance(review[name], str) or not review[name].strip():
            raise ContractError(f"review.{name}: expected nonempty string")
    if not isinstance(review["decisions"], list):
        raise ContractError("review.decisions: expected list")
    seen = set()
    accepted = 0
    for decision in review["decisions"]:
        _keys(decision, {"scene_id", "plausible", "note"}, "decision")
        sid = decision["scene_id"]
        if not isinstance(sid, str) or sid not in manifest["scene_ids"] or sid in seen:
            raise ContractError("decision.scene_id: unknown or duplicate scene")
        if type(decision["plausible"]) is not bool or not isinstance(decision["note"], str):
            raise ContractError("decision: plausible must be boolean and note must be text")
        seen.add(sid)
        accepted += decision["plausible"]
    if seen != set(manifest["scene_ids"]):
        raise ContractError("review.decisions: every gallery scene requires one decision")
    passed = manifest["count"] >= 50 and accepted == manifest["count"]
    return {"passed": passed, "status": "passed" if passed else "needs_fix", "count": len(seen), "plausible_count": accepted, "reviewer": review["reviewer"], "gallery_id": manifest["gallery_id"]}


COLORS = {"normal": "#455568", "shiny": "#ae6cbe", "black": "#151515", "glass": "#33a8be", "mirror": "#ed983e"}


def _svg(scene):
    grid = scene["grid"]
    res, origin, shape = grid["resolution"], grid["origin"], grid["shape"]
    width, height = shape[0] * res, shape[1] * res
    parts = [f'<svg role="img" aria-label="Floor plan" viewBox="{origin[0]-.2} {origin[1]-.2} {width+.4} {height+.4}" xmlns="http://www.w3.org/2000/svg">']
    dark = {r["room_id"] for r in scene["lighting"] if r["state"] == "dark"}
    for room in scene["rooms"]:
        points = " ".join(f"{x},{y}" for x, y in room["polygon"])
        parts.append(f'<polygon points="{points}" fill="{"#cbd1d8" if room["room_id"] in dark else "#f5f0df"}"/>')
    slice_z = math.floor((scene["spawn"][2] - origin[2]) / res)
    occupied = {tuple(v) for v in grid["occupied"]}
    # Include low furniture, but exclude floor/ceiling slabs from the projection.
    xy = {(x, y) for x, y, z in occupied if z == slice_z or (0 < z < slice_z)}
    materials = {(x, y): s["material"] for s in scene["surfaces"] for x, y, z in s["voxel_indices"] if (x,y) in xy and 0 < z <= slice_z}
    for x, y in sorted(xy):
        parts.append(f'<rect x="{origin[0]+x*res}" y="{origin[1]+y*res}" width="{res}" height="{res}" fill="{COLORS[materials.get((x,y), "normal")]}"/>')
    for door in scene["doors"]:
        for x, y in sorted({(x,y) for x,y,z in door["voxel_indices"] if z == slice_z}):
            parts.append(f'<rect x="{origin[0]+x*res}" y="{origin[1]+y*res}" width="{res}" height="{res}" fill="{"#dc5749" if door["state"] == "closed" else "#fff"}"/>')
    for room in scene["rooms"]:
        x = sum(p[0] for p in room["polygon"]) / len(room["polygon"])
        y = sum(p[1] for p in room["polygon"]) / len(room["polygon"])
        parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size=".23" fill="#101820" stroke="white" stroke-width=".035" paint-order="stroke">{html.escape(room["label"])}</text>')
    sx, sy = scene["spawn"][:2]
    parts.append(f'<g aria-label="Spawn location"><circle cx="{sx}" cy="{sy}" r=".22" fill="#fff" stroke="#0057d9" stroke-width=".08"/><path d="M {sx} {sy-.15} l -.09 .22 h .18 Z" fill="#0057d9"/></g>')
    parts.append(f'<path d="M {origin[0]+.4} {origin[1]+height-.4} h 1" stroke="#d92222" stroke-width=".07"/><text x="{origin[0]+.4}" y="{origin[1]+height-.55}" font-size=".2">1 m</text></svg>')
    return "".join(parts)


def build_gallery(scenes: list[dict], output_dir: Path, *, seed: int) -> dict:
    """Write a standalone gallery without making or defaulting human decisions."""
    if type(seed) is not int or not scenes:
        raise ContractError("gallery: integer seed and nonempty scenes required")
    for scene in scenes:
        validate_scene(scene)
    ids = [scene["scene_id"] for scene in scenes]
    if len(set(ids)) != len(ids):
        raise ContractError("gallery: duplicate scene_id")
    manifest = {"schema_version": "ares.floor_plan_gallery.v1", "count": len(scenes), "scene_ids": ids, "seed": seed, "source_scene_sha256": {s["scene_id"]: _digest(s) for s in scenes}}
    manifest["gallery_id"] = _digest(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gallery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    cards = []
    for i, scene in enumerate(scenes):
        sid = html.escape(scene["scene_id"])
        segments = scene["scene_id"].split(":")
        kind = segments[2] if len(segments) >= 4 and segments[0] == "world-v1" else scene["rooms"][0]["label"]
        cards.append(f'<article id="card{i}"><h2>{sid}</h2><p>Scene seed: {scene["seed"]} · Type: {html.escape(kind)}</p><button type="button" class="expand" aria-expanded="false" aria-controls="plan{i}">Expand plan</button><div id="plan{i}">{_svg(scene)}</div><fieldset><legend>Is this floor plan plausible?</legend><label><input type="radio" name="s{i}" value="true"> Plausible</label> <label><input type="radio" name="s{i}" value="false"> Needs fix</label><label class="note">Optional note<textarea id="n{i}"></textarea></label></fieldset></article>')
    data = json.dumps(manifest).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>ARES floor-plan human review</title><style>body{font:16px system-ui;margin:2rem;background:#f4f6f8;color:#17212d}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}article{background:white;padding:1rem;border:1px solid #bbc4cd;border-radius:8px}svg{display:block;width:100%;height:350px}article.expanded{grid-column:1/-1}article.expanded svg{height:80vh}h2{font-size:1rem;overflow-wrap:anywhere}label{display:inline-block;margin:.4rem}textarea{display:block;width:95%;min-height:3rem}.note{display:block}button,input{font:inherit}button{padding:.7rem}#status{font-weight:bold}</style><h1>Human floor-plan review</h1><p>Inspect every plan for plausible rooms, connections, circulation and furniture. No decisions are preselected. Grey rooms are dark; white door cells are openings; red door cells are closed. Walls and furniture show occupied footprints above the floor through spawn height. Labels give room types. Each plan has a 1 metre scale; a blue arrow in a white circle marks the spawn location. Expand a plan to inspect labels and circulation.</p><p>Materials: <span style="color:#455568">normal</span> · <span style="color:#ae6cbe">shiny</span> · black · <span style="color:#33a8be">glass</span> · <span style="color:#ed983e">mirror</span>.</p><label>Reviewer <input id="reviewer" required></label><label>Review date/time <input id="date" required placeholder="YYYY-MM-DD and time"></label><button id="export">Export completed review JSON</button><p id="status" role="status"></p><main>''' + "".join(cards) + '''</main><script>
document.querySelectorAll('.expand').forEach(button=>button.onclick=()=>{const expanded=button.closest('article').classList.toggle('expanded');button.setAttribute('aria-expanded',String(expanded));button.textContent=expanded?'Collapse plan':'Expand plan';});
const manifest=DATA; const storageKey='ares-review-'+manifest.gallery_id;
function draft(){return {reviewer:document.getElementById('reviewer').value,reviewed_at:document.getElementById('date').value,decisions:manifest.scene_ids.map((scene_id,i)=>({scene_id,plausible:document.querySelector('input[name="s'+i+'"]:checked')?.value,note:document.getElementById('n'+i).value}))};}
try {const saved=JSON.parse(localStorage.getItem(storageKey));if(saved){document.getElementById('reviewer').value=saved.reviewer||'';document.getElementById('date').value=saved.reviewed_at||'';saved.decisions.forEach((d,i)=>{if(d.scene_id===manifest.scene_ids[i]){if(d.plausible==='true'||d.plausible==='false')document.querySelector('input[name="s'+i+'"][value="'+d.plausible+'"]').checked=true;document.getElementById('n'+i).value=d.note||'';}});}}catch(e){}
document.addEventListener('input',()=>{try{localStorage.setItem(storageKey,JSON.stringify(draft()));}catch(e){}});
document.getElementById('export').onclick=()=>{const d=draft();const status=document.getElementById('status');if(!d.reviewer.trim()||!d.reviewed_at.trim()||d.decisions.some(v=>v.plausible===undefined)){status.textContent='Enter reviewer and date/time, and decide every plan.';return;}d.decisions.forEach(v=>v.plausible=v.plausible==='true');const result={schema_version:'ares.floor_plan_review.v1',gallery_id:manifest.gallery_id,source_scene_sha256:manifest.source_scene_sha256,...d};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='floor_plan_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='Review exported. Submit the JSON to the Phase 1 CLI.';};
</script></html>'''.replace('DATA', data)
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    return manifest
