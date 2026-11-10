//! Hex-dominant meshing of a design body or an imported surface.

mod policy;

use fieldcraft_mesh::cdt::CdtOptions;
use fieldcraft_mesh::dominant::{MAX_SOURCE_TETS, Recombined, recombine};
use fieldcraft_mesh::hex::HexMethod;
use fieldcraft_mesh::octree::OctreeOptions;
use fieldcraft_model::{ElementKind, GeometryRef, Model, progress};
use policy::{Partition, Policy};
use serde_json::{Value, json};

use super::hex::{DEFAULT_RELATIVE_SIZE, MIN_RELATIVE_SIZE, add_hex, hex_body, hex_body_with_limit, hex_body_with_section_guide, inputs};
use super::mesh::{BodyTets, LOW_SCALED_JACOBIAN, mesh_body, optional_count, quality_options};
use super::octree::{MIN_RELATIVE_SIZE as OCTREE_MIN_RELATIVE_SIZE, add_octree, octree_body};
use crate::requirement::Requirement;
use crate::{CommandKind, CommandSpec, EngineError, Result, Session};

pub const HEX_DOMINANT: CommandSpec = CommandSpec {
    id: "mesh.hex.dominant",
    label: "Hex-Dominant Mesh",
    params: r#"{"body": "Body1" (default: every design body) or "surface": 0, "size": 2.0 (target length), "tolerance": 0.05 (chordal length), "max_tets": 500000 (source cap), "requirements"?: {"min_scaled_jacobian": 0.2, "max_edge_ratio": 100, "min_hex_volume_fraction": 0.5, "partition": "auto" | "standard" | "alternate"}, plus "weld", "weld_distance", "feature_angle"} — with requirements, commit all selected bodies only when every body passes. Use fraction 1 for all Hex8. Auto tries standard and alternate extrusion cuts at the same CAD tolerance, then octree and mixed routes. An explicit partition limits the search to that structured candidate. Replies outcome (committed or unmet), applied limits, each body's chosen method and quality, rejected candidates with codes and measurements, and next_attempt when another partition is available. An unmet result keeps the existing mesh. Automatic relaxation fields are skipped for the checked result. Without requirements, uses the standard structured, octree and tet-recombination sequence and reports low quality as a warning."#,
    form: Some(super::forms::HEX_DOMINANT),
    kind: CommandKind::Model,
    requires: &[Requirement::Geometry],
    run: hex_dominant,
};

/// Fraction of the input boundary's enclosed volume.
const MAX_HEX_VOLUME_ERROR: f64 = 0.02;
const STRUCTURED_REFINEMENTS: usize = 3;
const MAX_AUTOMATIC_HEX_ELEMENTS: usize = 250_000;

fn volume_error(volume: f64, surface: f64) -> f64 {
    ((volume - surface) / surface).abs()
}

fn mesh_kind(e: &EngineError) -> Option<fieldcraft_mesh::Kind> {
    match e {
        EngineError::Context { source, .. } => mesh_kind(source),
        EngineError::Mesh(e) => e.kind(),
        _ => None,
    }
}

fn recoverable(e: &EngineError) -> bool {
    use fieldcraft_mesh::Kind;
    matches!(mesh_kind(e), Some(Kind::Unsupported | Kind::Failed | Kind::Limit))
}

fn failed_attempt(s: &Session, route: &str, e: EngineError) -> Value {
    use fieldcraft_mesh::Kind;
    let code = match mesh_kind(&e) {
        Some(Kind::Unsupported) => "unsupported_geometry",
        Some(Kind::Limit) => "work_limit",
        Some(Kind::Surface) => "invalid_boundary",
        _ => "generation_failed",
    };
    json!({"method": route, "code": code, "reason": e.shown(crate::units::geometry(), s.units()).to_string()})
}

fn rejected_volume(route: &str, error: f64) -> Value {
    json!({"method": route, "reason": "volume_error", "code": "volume_error", "volume_error_fraction": error, "limit": MAX_HEX_VOLUME_ERROR})
}

fn report_quality(min: f64) -> Value {
    json!({"min_scaled_jacobian": min, "low_scaled_jacobian": min < LOW_SCALED_JACOBIAN})
}

fn unmet(body: &str, rejected: Vec<Value>, retry_partition: Option<&str>) -> Value {
    let mut report = json!({"body":body,"outcome":"unmet","rejected":rejected});
    if let Some(partition) = retry_partition {
        report["next_attempt"] = json!({
            "command":"mesh.hex.dominant",
            "params_patch":{"requirements":{"partition":partition}},
        });
    }
    report
}

fn hex_dominant(s: &mut Session, p: &Value) -> Result<Value> {
    let policy = Policy::parse(p)?;
    let cap = s.limits.max_tets.min(MAX_SOURCE_TETS);
    let max_tets = optional_count(p, "max_tets", cap as u64, 1..=MAX_SOURCE_TETS as u64)?.min(cap);
    let (mut options, _) = quality_options(p, max_tets as u64)?;
    options.max_tets = max_tets;
    if options.order != fieldcraft_mesh::tet::Order::Linear {
        return Err(EngineError::Params("Hex-Dominant Mesh creates linear elements".into()));
    }
    let cdt = CdtOptions { max_tets, max_steiner: s.limits.max_steiner };
    let bodies = inputs(s, p)?;
    let several = bodies.len() > 1;
    let sizing = super::sizes::sizing(s)?;
    let mut model = s.model.clone();
    let cleared = super::replace::clear(&mut model, bodies.iter().map(|b| b.name.as_str()));
    let mut report = Vec::with_capacity(bodies.len());
    let mut octree_budget = s.limits.max_hex_elements.min(MAX_AUTOMATIC_HEX_ELEMENTS);
    let hex_limit = octree_budget;
    'body: for b in bodies {
        progress::check()?;
        let mut rejected = Vec::new();
        let initial_size = options.size.unwrap_or(DEFAULT_RELATIVE_SIZE * b.diagonal);
        let mut size = initial_size;
        let seeded = s.model.seeds().iter().any(|seed| seed.edge.body == b.name);
        let guide = if seeded || p.get("surface").is_some() { None } else { s.design.as_deref().and_then(|d| d.body(&b.name)) };
        let partitions = policy.as_ref().map_or(&[Partition::Standard][..], |q| q.partitions);
        for refinement in 0..=STRUCTURED_REFINEMENTS {
            let mut volume_failed = false;
            for &partition in partitions {
                progress::check()?;
                progress::stage(&format!("{}: {} structured partition", b.name, partition.name()), None, "");
                let attempt = match partition {
                    Partition::Standard if refinement == 0 && policy.is_none() => hex_body(p, &b, HexMethod::Auto, false, Some(size), &sizing),
                    Partition::Standard => {
                        let cap = if policy.is_some() { hex_limit } else { MAX_AUTOMATIC_HEX_ELEMENTS };
                        hex_body_with_limit(p, &b, HexMethod::Auto, false, Some(size), &sizing, cap)
                    }
                    Partition::Alternate if refinement != 0 => continue,
                    Partition::Alternate => {
                        let Some(shape) = guide else {
                            rejected.push(json!({"method":"extrude","partition":"alternate","code":"alternate_unavailable",
                                "reason":if seeded { "edge_seeds" } else { "needs_cad_body" }}));
                            continue;
                        };
                        hex_body_with_section_guide(p, &b, size, &sizing, hex_limit, shape)
                    }
                };
                let m = match attempt {
                    Ok(mesh) => mesh,
                    Err(e)
                        if recoverable(&e)
                            || (policy.is_some()
                                && !e.is_cancelled()
                                && (mesh_kind(&e).is_some() || matches!(e, EngineError::Geometry(_)) && partition == Partition::Alternate)) =>
                    {
                        let mut failure = failed_attempt(s, "structured", e);
                        failure["partition"] = json!(partition.name());
                        rejected.push(failure);
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                let r = &m.report;
                let error = volume_error(r.volume, r.surface_volume);
                if error > MAX_HEX_VOLUME_ERROR || !error.is_finite() {
                    let mut failure = rejected_volume(r.method.name(), error);
                    failure["partition"] = json!(partition.name());
                    rejected.push(failure);
                    volume_failed |= partition == Partition::Standard;
                    continue;
                }
                let mut quality = report_quality(r.min_scaled_jacobian);
                if let Some(policy) = &policy {
                    let check = policy.inspect(&m.nodes, m.connectivity.chunks_exact(m.kind.node_count()).map(|ids| (m.kind, ids)))?;
                    if !check.violations.is_empty() {
                        rejected.push(json!({"method":r.method.name(),"partition":partition.name(),"code":"quality_unmet",
                            "quality":check.quality,"violations":check.violations}));
                        continue;
                    }
                    quality = check.quality;
                }
                add_hex(&mut model, &b, &m)?;
                report.push(json!({
                    "body": b.name, "method": r.method.name(), "partition": partition.name(), "hex_dominant": true,
                    "hex8": m.element_count(), "pyramid5": 0, "tet4": 0, "hex_volume_fraction": 1.0,
                    "elements": m.element_count(), "nodes": m.nodes.len(), "boundary_faces": m.boundary.len(),
                    "volume": r.volume, "surface_volume": r.surface_volume, "volume_error_fraction": error,
                    "quality": quality, "size": sizing.body_size(&b.name, size),
                    "structured_refinements": refinement, "rejected": rejected,
                }));
                continue 'body;
            }
            if !volume_failed {
                break;
            }
            let finer = size * 0.5;
            if finer < MIN_RELATIVE_SIZE * b.diagonal || sizing.body_size(&b.name, finer) >= sizing.body_size(&b.name, size) {
                break;
            }
            size = finer;
        }
        if policy.is_some() && partitions.len() == 1 {
            let retry = (guide.is_some() && partitions == [Partition::Standard]).then_some(if several { "auto" } else { "alternate" });
            report.push(unmet(&b.name, rejected, retry));
            continue;
        }
        let octree_min_size = OCTREE_MIN_RELATIVE_SIZE * b.diagonal;
        let octree_size = sizing.body_size(&b.name, initial_size);
        let skip_octree = if octree_budget == 0 {
            Some("element_budget")
        } else if seeded {
            Some("edge_seeds")
        } else if octree_size < octree_min_size {
            Some("size_below_octree_minimum")
        } else {
            None
        };
        if let Some(reason) = skip_octree {
            rejected.push(json!({"method": "octree", "code": reason, "reason": reason}));
        } else {
            let octree = OctreeOptions { max_elements: octree_budget, attempts: 2, ..OctreeOptions::default() };
            // A small change of spacing changes which grid cells connect through narrow features.
            let finer = initial_size * 0.75;
            let refine_octree = finer >= octree_min_size && sizing.body_size(&b.name, finer) < octree_size;
            for size in [Some(initial_size), refine_octree.then_some(finer)].into_iter().flatten() {
                match octree_body(p, &b, Some(size), &sizing, &octree) {
                    Ok(m) => {
                        let r = &m.report;
                        let error = volume_error(r.volume, r.surface_volume);
                        if error <= MAX_HEX_VOLUME_ERROR {
                            let mut quality = report_quality(r.min_scaled_jacobian);
                            if let Some(policy) = &policy {
                                let check = policy.inspect(&m.nodes, m.hexes.iter().map(|ids| (ElementKind::Hex8, ids.as_slice())))?;
                                if !check.violations.is_empty() {
                                    rejected.push(
                                        json!({"method":"octree","code":"quality_unmet","quality":check.quality,"violations":check.violations}),
                                    );
                                    continue;
                                }
                                quality = check.quality;
                            }
                            add_octree(&mut model, &b, &m)?;
                            octree_budget = octree_budget.saturating_sub(m.hexes.len());
                            report.push(json!({
                                "body": b.name, "method": "octree", "hex_dominant": true,
                                "hex8": m.hexes.len(), "pyramid5": 0, "tet4": 0, "hex_volume_fraction": 1.0,
                                "elements": m.hexes.len(), "nodes": m.nodes.len(), "boundary_faces": m.boundary.len(),
                                "volume": r.volume, "surface_volume": r.surface_volume, "volume_error_fraction": error,
                                "size": sizing.body_size(&b.name, size),
                                "quality": quality, "rejected": rejected,
                            }));
                            continue 'body;
                        }
                        rejected.push(rejected_volume("octree", error));
                    }
                    Err(e) if recoverable(&e) || (policy.is_some() && !e.is_cancelled() && mesh_kind(&e).is_some()) => {
                        let retry = mesh_kind(&e) == Some(fieldcraft_mesh::Kind::Failed);
                        rejected.push(failed_attempt(s, "octree", e));
                        if !retry {
                            break;
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        if policy.as_ref().is_some_and(|p| p.min_hex_fraction == 1.0) {
            rejected.push(json!({"method":"tet_recombined","code":"mixed_not_allowed","reason":"all_hex_required"}));
            report.push(unmet(&b.name, rejected, None));
            continue;
        }
        let body_options = sizing.tet_options(&b.name, p, &options);
        let body_name = b.name.clone();
        let b = match mesh_body(s.design.as_deref(), p, b, &cdt, &body_options, true, several) {
            Ok(body) => body,
            Err(e) if policy.is_some() && !e.is_cancelled() && mesh_kind(&e).is_some() => {
                rejected.push(failed_attempt(s, "tet", e));
                report.push(unmet(&body_name, rejected, None));
                continue;
            }
            Err(e) => return Err(e),
        };
        progress::stage(&format!("{}: recombine tetrahedra", b.name), None, "");
        let m = recombine(&b.mesh).map_err(|e| EngineError::of_body(&b.name, e))?;
        let volume = b.mesh.report.volume;
        let fraction = (m.hex_volume / volume).clamp(0.0, 1.0);
        let error = volume_error(volume, b.surface_volume);
        let mut quality = if m.hexes == 0 { report_quality(b.mesh.report.output.min_scaled_jacobian) } else { Value::Null };
        if let Some(policy) = &policy {
            let mut check = policy.inspect(&b.mesh.points, m.elements.iter().map(|(kind, ids)| (*kind, ids.as_slice())))?;
            if let Some(violation) = policy.fraction_violation(fraction, m.hexes == m.elements.len()) {
                check.violations.push(violation);
            }
            if error > MAX_HEX_VOLUME_ERROR || !error.is_finite() {
                check.violations.push(json!({"code":"volume_error","actual":error,"limit":MAX_HEX_VOLUME_ERROR}));
            }
            if !check.violations.is_empty() {
                rejected.push(json!({"method":"tet_recombined","code":"quality_unmet","quality":check.quality,"violations":check.violations}));
                report.push(unmet(&b.name, rejected, None));
                continue;
            }
            quality = check.quality;
        }
        add_mixed(&mut model, &b, &m)?;
        report.push(json!({
            "body": b.name, "method": if m.hexes == 0 { "tet" } else { "recombined" }, "hex_dominant": fraction > 0.5,
            "hex8": m.hexes, "pyramid5": m.pyramids, "tet4": m.elements.len() - m.hexes - m.pyramids,
            "hex_volume_fraction": fraction,
            "elements": m.elements.len(), "nodes": b.mesh.points.len(), "boundary_faces": m.boundary.len(),
            "volume": volume, "surface_volume": b.surface_volume, "volume_error_fraction": error,
            "quality": quality,
            "rejected": rejected,
        }));
    }
    progress::check()?;
    if let Some(policy) = &policy
        && report.iter().any(|b| b["outcome"] == "unmet")
    {
        return Ok(json!({"outcome":"unmet","committed":false,"requirements":policy.limits(),"bodies":report,
            "nodes":s.model.nodes().len(),"elements":s.model.elements().len(),"summary":"mesh requirements unmet; kept existing meshes"}));
    }
    let (nodes, elements) = (model.nodes().len(), model.elements().len());
    let without_majority = report.iter().filter(|b| b["hex_dominant"] == false).count();
    let low_quality = report.iter().filter(|b| b["quality"]["low_scaled_jacobian"] == true).count();
    let mut reply = json!({"bodies": report, "nodes": nodes, "elements": elements});
    if let Some(policy) = &policy {
        reply["outcome"] = json!("committed");
        reply["committed"] = json!(true);
        reply["requirements"] = policy.limits();
        if model.relaxations().values().any(|f| f.reapply) {
            reply["automatic_relaxation"] = json!("skipped_for_requirements");
        }
    }
    let summary = match (without_majority, low_quality) {
        (0, 0) => None,
        (n, 0) => Some(format!("{n} of {} bodies have no hex majority", report.len())),
        (0, n) => Some(format!("{n} of {} bodies have a scaled Jacobian below {LOW_SCALED_JACOBIAN}", report.len())),
        (n, q) => Some(format!("{n} of {} bodies have no hex majority, {q} have a scaled Jacobian below {LOW_SCALED_JACOBIAN}", report.len())),
    };
    if let Some(summary) = summary {
        reply["summary"] = json!(summary);
    }
    s.model = model;
    Ok(cleared.report(s, reply))
}

fn add_mixed(model: &mut Model, body: &BodyTets, mesh: &Recombined) -> Result<()> {
    let (first_node, first_element) = super::fresh_labels(model, body.mesh.points.len(), mesh.elements.len())?;
    for (i, &point) in body.mesh.points.iter().enumerate() {
        model.set_node(first_node + i as u32, point)?;
    }
    let tag = model.geometry_tag(&GeometryRef::body(&body.name)?)?;
    for (i, (kind, nodes)) in mesh.elements.iter().enumerate() {
        let label = first_element + i as u32;
        model.set_element(label, *kind, nodes.iter().map(|v| first_node + v).collect())?;
        model.tag_element(label, tag)?;
    }
    for &(element, face, tag) in &mesh.boundary {
        let name = body.face_names.get(tag as usize).ok_or_else(|| EngineError::Params(format!("boundary face {tag} has no geometry face")))?;
        let tag = model.geometry_tag(&GeometryRef::face(&body.name, name.clone())?)?;
        model.tag_element_face(first_element + element as u32, face, tag)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fieldcraft_geometry::sample::{Plate, plate_design};
    use fieldcraft_geometry::{SurfaceMesh, open_design, script_design, tagged_mesh::TaggedSurface};

    use super::*;

    #[test]
    fn a_sweepable_plate_reports_only_hexes() {
        let plate = Plate::default();
        let mut s = Session { design: Some(Arc::new(open_design(&plate_design(&plate).unwrap()).unwrap())), ..Session::default() };
        let p = json!({"body": "Body1", "size": 5.0});
        let r = s.execute("mesh.hex.dominant", &p).unwrap();
        assert!(r["bodies"][0]["hex8"].as_u64().unwrap() > 0);
        assert_eq!(r["bodies"][0]["hex_volume_fraction"], 1.0);
        assert_eq!(r["bodies"][0]["tet4"], 0);
    }

    #[test]
    fn unmet_requirements_keep_the_published_mesh_and_explain_the_next_attempt() {
        let plate = Plate::default();
        let mut s = Session { design: Some(Arc::new(open_design(&plate_design(&plate).unwrap()).unwrap())), ..Session::default() };
        let cmd = "mesh.hex.dominant";
        let r = s.execute(cmd, &json!({"body":"Body1","size":5,"requirements":{"min_hex_volume_fraction":1.0}})).unwrap();
        assert_eq!(r["outcome"], "committed");
        let published = s.model.clone();
        let r = s.execute(cmd, &json!({"body":"Body1","size":5,"requirements":{"partition":"standard","max_edge_ratio":1.0}})).unwrap();
        assert_eq!(r["outcome"], "unmet");
        assert_eq!(r["committed"], false);
        let b = &r["bodies"][0];
        assert_eq!(b["rejected"][0]["code"], "quality_unmet");
        assert_eq!(b["rejected"][0]["violations"][0]["code"], "max_edge_ratio");
        assert!(b["rejected"][0]["violations"][0]["actual"].as_f64().unwrap() > 1.0);
        assert_eq!(b["next_attempt"]["params_patch"]["requirements"]["partition"], "alternate");
        assert_eq!(s.model, published);
        assert!(s.execute(cmd, &json!({"body":"Body1","requirements":{"max_edge_rato":10.0}})).is_err());
        assert_eq!(s.model, published);
    }

    fn recipe(script: &str) -> Session {
        Session { design: Some(Arc::new(open_design(&script_design(script).unwrap()).unwrap())), ..Session::default() }
    }

    #[test]
    fn the_bent_tube_refines_until_the_hex_mesh_preserves_volume() {
        let script = include_str!("../../../../tests/corpus/bent-tube/recipe.json");
        let mut s = recipe(script);
        for size in [4, 8] {
            let r = s.execute("mesh.hex.dominant", &json!({"size": size})).unwrap();
            let b = &r["bodies"][0];
            assert_eq!(b["method"], "sweep");
            assert_eq!(b["hex_dominant"], true);
            assert!(b["structured_refinements"].as_u64().unwrap() > 0);
            assert!(b["volume_error_fraction"].as_f64().unwrap() <= MAX_HEX_VOLUME_ERROR);
            assert_eq!(b["rejected"][0]["reason"], "volume_error");
        }
    }

    #[test]
    fn the_filleted_bracket_uses_octree_after_structured_hex_is_refused() {
        let script = include_str!("../../../../tests/corpus/filleted-bracket/recipe.json");
        let mut s = recipe(script);
        let r = s.execute("mesh.hex.dominant", &json!({"size": 6})).unwrap();
        let b = &r["bodies"][0];
        assert_eq!(b["method"], "octree");
        assert_eq!(b["hex_volume_fraction"], 1.0);
        assert!(b["volume_error_fraction"].as_f64().unwrap() <= MAX_HEX_VOLUME_ERROR);
        assert_eq!(b["rejected"][0]["method"], "structured");
        assert!(s.model.elements().values().all(|e| e.kind == fieldcraft_model::ElementKind::Hex8));
        assert!(s.model.classification().entities().contains(&GeometryRef::face("Bracket", "F2:side:l1").unwrap()));
    }

    #[test]
    fn the_sharp_wedge_reports_low_quality_despite_passing_the_volume_check() {
        let script = include_str!("../../../../tests/corpus/sharp-wedge/recipe.json");
        let mut s = recipe(script);
        let r = s.execute("mesh.hex.dominant", &json!({"size": 4})).unwrap();
        let b = &r["bodies"][0];
        assert_eq!(b["hex_dominant"], true);
        assert_eq!(b["quality"]["low_scaled_jacobian"], true);
        assert!(r["summary"].as_str().unwrap().contains("scaled Jacobian"));
    }

    #[test]
    #[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
    fn the_sphere_on_plate_uses_a_finer_octree_when_the_initial_grid_does_not_fit() {
        let script = include_str!("../../../../tests/corpus/sphere-on-plate/recipe.json");
        let mut s = recipe(script);
        let r = s.execute("mesh.hex.dominant", &json!({"size": 4})).unwrap();
        let b = &r["bodies"][0];
        assert_eq!(b["method"], "octree");
        assert_eq!(b["hex_dominant"], true);
        assert!(b["size"].as_f64().unwrap() <= 4.0);
        assert!(b["volume_error_fraction"].as_f64().unwrap() <= MAX_HEX_VOLUME_ERROR);
    }

    #[test]
    fn an_imported_tetrahedron_reports_zero_hexes_and_keeps_its_boundary() {
        let mesh = SurfaceMesh {
            positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            triangles: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
            tri_face: vec![0, 1, 2, 3],
            ..SurfaceMesh::default()
        };
        let mut s = Session::default();
        s.limits.max_hex_elements = 0;
        s.surfaces.push(Arc::new(TaggedSurface { name: "Tet".into(), mesh, faces: 4, original: None }));
        let r = s.execute("mesh.hex.dominant", &json!({"surface": 0, "size": 2000.0})).unwrap();
        let b = &r["bodies"][0];
        assert_eq!(b["method"], "tet");
        assert_eq!(b["hex_volume_fraction"], 0.0);
        assert_eq!(b["hex_dominant"], false);
        assert!(r["summary"].as_str().unwrap().contains("no hex majority"));
        assert_eq!(b["tet4"], 1);
        assert_eq!(b["boundary_faces"], 4);
        assert!(s.execute("mesh.hex.dominant", &json!({"surface": 0, "size": 2000.0, "max_tets": 1})).is_err());
        assert_eq!(s.model.elements().len(), 1);
        let old = s.model.clone();
        let r = s.execute("mesh.hex.dominant", &json!({"surface":0,"size":2000,"requirements":{"min_hex_volume_fraction":1.0}})).unwrap();
        assert_eq!(r["outcome"], "unmet");
        let attempts = r["bodies"][0]["rejected"].as_array().unwrap();
        assert_eq!(attempts.last().unwrap()["code"], "mixed_not_allowed");
        let r = s.execute("mesh.hex.dominant", &json!({"surface":0,"size":2000,"requirements":{"min_hex_volume_fraction":0.5}})).unwrap();
        assert_eq!(r["outcome"], "unmet");
        let attempts = r["bodies"][0]["rejected"].as_array().unwrap();
        assert!(attempts.last().unwrap()["violations"].as_array().unwrap().iter().any(|v| v["code"] == "min_hex_volume_fraction"));
        assert_eq!(s.model, old);
    }
}
