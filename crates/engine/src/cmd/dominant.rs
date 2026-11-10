//! Hex-dominant meshing of a design body or an imported surface.

use fieldcraft_mesh::cdt::CdtOptions;
use fieldcraft_mesh::dominant::{MAX_SOURCE_TETS, Recombined, recombine};
use fieldcraft_mesh::hex::HexMethod;
use fieldcraft_mesh::octree::OctreeOptions;
use fieldcraft_model::{GeometryRef, Model, progress};
use serde_json::{Value, json};

use super::hex::{DEFAULT_RELATIVE_SIZE, MIN_RELATIVE_SIZE, add_hex, hex_body, hex_body_with_limit, inputs};
use super::mesh::{BodyTets, LOW_SCALED_JACOBIAN, mesh_body, optional_count, quality_options};
use super::octree::{MIN_RELATIVE_SIZE as OCTREE_MIN_RELATIVE_SIZE, add_octree, octree_body};
use crate::requirement::Requirement;
use crate::{CommandKind, CommandSpec, EngineError, Result, Session};

pub const HEX_DOMINANT: CommandSpec = CommandSpec {
    id: "mesh.hex.dominant",
    label: "Hex-Dominant Mesh",
    params: r#"{"body": "Body1" (default: every design body) or "surface": 0 (imported surface), "size": 2.0 (target element size, a length; default from the body), "tolerance": 0.05 (chordal tolerance, a length), "max_tets": 500000 (limit on the source tetrahedra, also capped by the session), and the surface options "weld", "weld_distance", "feature_angle"} — replaces the selected meshes with linear solids. Tries structured hex8, halving the target size up to three times with a cap of 250000 elements per retry if the volume differs from the triangulated input by more than 2%. Then tries a bounded octree mesh on unseeded bodies with the same volume check, with one finer grid size after a fit or volume failure. Otherwise makes a tet4 mesh and recombines planar six-tet fans with conforming pyramid5 transitions. Reports each body's method, rejected attempts, volume error and hex_volume_fraction. hex_dominant is true when hexes fill more than half the volume. Boundary faces retain their geometry tags."#,
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
    json!({"method": route, "reason": e.shown(crate::units::geometry(), s.units()).to_string()})
}

fn rejected_volume(route: &str, error: f64) -> Value {
    json!({"method": route, "reason": "volume_error", "volume_error_fraction": error})
}

fn report_quality(min: f64) -> Value {
    json!({"min_scaled_jacobian": min, "low_scaled_jacobian": min < LOW_SCALED_JACOBIAN})
}

fn hex_dominant(s: &mut Session, p: &Value) -> Result<Value> {
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
    'body: for b in bodies {
        let mut rejected = Vec::new();
        let initial_size = options.size.unwrap_or(DEFAULT_RELATIVE_SIZE * b.diagonal);
        let mut size = initial_size;
        for refinement in 0..=STRUCTURED_REFINEMENTS {
            let structured = if refinement == 0 {
                hex_body(p, &b, HexMethod::Auto, false, Some(size), &sizing)
            } else {
                hex_body_with_limit(p, &b, HexMethod::Auto, false, Some(size), &sizing, MAX_AUTOMATIC_HEX_ELEMENTS)
            };
            match structured {
                Ok(m) => {
                    let r = &m.report;
                    let error = volume_error(r.volume, r.surface_volume);
                    if error <= MAX_HEX_VOLUME_ERROR {
                        add_hex(&mut model, &b, &m)?;
                        report.push(json!({
                            "body": b.name, "method": r.method.name(), "hex_dominant": true,
                            "hex8": m.element_count(), "pyramid5": 0, "tet4": 0, "hex_volume_fraction": 1.0,
                            "elements": m.element_count(), "nodes": m.nodes.len(), "boundary_faces": m.boundary.len(),
                            "volume": r.volume, "surface_volume": r.surface_volume, "volume_error_fraction": error,
                            "quality": report_quality(r.min_scaled_jacobian),
                            "size": sizing.body_size(&b.name, size), "structured_refinements": refinement, "rejected": rejected,
                        }));
                        continue 'body;
                    }
                    rejected.push(rejected_volume(r.method.name(), error));
                }
                Err(e) if recoverable(&e) => {
                    rejected.push(failed_attempt(s, "structured", e));
                    break;
                }
                Err(e) => return Err(e),
            }
            let finer = size * 0.5;
            if finer < MIN_RELATIVE_SIZE * b.diagonal || sizing.body_size(&b.name, finer) >= sizing.body_size(&b.name, size) {
                break;
            }
            size = finer;
        }
        let octree_min_size = OCTREE_MIN_RELATIVE_SIZE * b.diagonal;
        let octree_size = sizing.body_size(&b.name, initial_size);
        let skip_octree = if octree_budget == 0 {
            Some("element_budget")
        } else if s.model.seeds().iter().any(|seed| seed.edge.body == b.name) {
            Some("edge_seeds")
        } else if octree_size < octree_min_size {
            Some("size_below_octree_minimum")
        } else {
            None
        };
        if let Some(reason) = skip_octree {
            rejected.push(json!({"method": "octree", "reason": reason}));
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
                            add_octree(&mut model, &b, &m)?;
                            octree_budget = octree_budget.saturating_sub(m.hexes.len());
                            report.push(json!({
                                "body": b.name, "method": "octree", "hex_dominant": true,
                                "hex8": m.hexes.len(), "pyramid5": 0, "tet4": 0, "hex_volume_fraction": 1.0,
                                "elements": m.hexes.len(), "nodes": m.nodes.len(), "boundary_faces": m.boundary.len(),
                                "volume": r.volume, "surface_volume": r.surface_volume, "volume_error_fraction": error,
                                "size": sizing.body_size(&b.name, size),
                                "quality": report_quality(r.min_scaled_jacobian), "rejected": rejected,
                            }));
                            continue 'body;
                        }
                        rejected.push(rejected_volume("octree", error));
                    }
                    Err(e) if recoverable(&e) => {
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
        let body_options = sizing.tet_options(&b.name, p, &options);
        let b = mesh_body(s.design.as_deref(), p, b, &cdt, &body_options, true, several)?;
        progress::stage(&format!("{}: recombine tetrahedra", b.name), None, "");
        let m = recombine(&b.mesh).map_err(|e| EngineError::of_body(&b.name, e))?;
        add_mixed(&mut model, &b, &m)?;
        let volume = b.mesh.report.volume;
        let fraction = (m.hex_volume / volume).clamp(0.0, 1.0);
        report.push(json!({
            "body": b.name, "method": if m.hexes == 0 { "tet" } else { "recombined" }, "hex_dominant": fraction > 0.5,
            "hex8": m.hexes, "pyramid5": m.pyramids, "tet4": m.elements.len() - m.hexes - m.pyramids,
            "hex_volume_fraction": fraction,
            "elements": m.elements.len(), "nodes": b.mesh.points.len(), "boundary_faces": m.boundary.len(),
            "volume": volume, "surface_volume": b.surface_volume, "volume_error_fraction": volume_error(volume, b.surface_volume),
            "quality": if m.hexes == 0 { report_quality(b.mesh.report.output.min_scaled_jacobian) } else { Value::Null },
            "rejected": rejected,
        }));
    }
    let (nodes, elements) = (model.nodes().len(), model.elements().len());
    let without_majority = report.iter().filter(|b| b["hex_dominant"] == false).count();
    let low_quality = report.iter().filter(|b| b["quality"]["low_scaled_jacobian"] == true).count();
    let mut reply = json!({"bodies": report, "nodes": nodes, "elements": elements});
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
    }
}
