//! Hex-dominant meshing of a design body or an imported surface.

use fieldcraft_mesh::cdt::CdtOptions;
use fieldcraft_mesh::dominant::{MAX_SOURCE_TETS, Recombined, recombine};
use fieldcraft_mesh::hex::HexMethod;
use fieldcraft_model::{GeometryRef, Model, progress};
use serde_json::{Value, json};

use super::hex::{add_hex, hex_body, inputs};
use super::mesh::{BodyTets, mesh_body, optional_count, quality_options};
use crate::requirement::Requirement;
use crate::{CommandKind, CommandSpec, EngineError, Result, Session};

pub const HEX_DOMINANT: CommandSpec = CommandSpec {
    id: "mesh.hex.dominant",
    label: "Hex-Dominant Mesh",
    params: r#"{"body": "Body1" (default: every design body) or "surface": 0 (imported surface), "size": 2.0 (element size, a length; default from the body), "tolerance": 0.05 (chordal tolerance, a length), "max_tets": 500000 (limit on the source tetrahedra, also capped by the session), and the surface options "weld", "weld_distance", "feature_angle"} — replaces the selected meshes with linear solids. Uses structured hex8 when it fits. Otherwise it makes a tet4 mesh and recombines six-tet fans with planar faces, using pyramid5 where two remaining tetrahedra can form a conforming transition. Reports each body's counts and hex_volume_fraction (0 when no hex could be formed). Boundary faces retain their geometry tags."#,
    form: Some(super::forms::HEX_DOMINANT),
    kind: CommandKind::Model,
    requires: &[Requirement::Geometry],
    run: hex_dominant,
};

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
    for b in bodies {
        match hex_body(p, &b, HexMethod::Auto, false, options.size, &sizing) {
            Ok(m) => {
                add_hex(&mut model, &b, &m)?;
                report.push(json!({
                    "body": b.name, "method": m.report.method.name(),
                    "hex8": m.element_count(), "pyramid5": 0, "tet4": 0, "hex_volume_fraction": 1.0,
                    "elements": m.element_count(), "nodes": m.nodes.len(), "boundary_faces": m.boundary.len(),
                    "volume": m.report.volume,
                }));
                continue;
            }
            Err(EngineError::Context { source, .. }) if matches!(*source, EngineError::Mesh(ref e) if matches!(e.kind(), Some(fieldcraft_mesh::Kind::Unsupported | fieldcraft_mesh::Kind::Failed))) =>
                {}
            Err(e) => return Err(e),
        }
        let body_options = sizing.tet_options(&b.name, p, &options);
        let b = mesh_body(s.design.as_deref(), p, b, &cdt, &body_options, true, several)?;
        progress::stage(&format!("{}: recombine tetrahedra", b.name), None, "");
        let m = recombine(&b.mesh).map_err(|e| EngineError::of_body(&b.name, e))?;
        add_mixed(&mut model, &b, &m)?;
        let volume = b.mesh.report.volume;
        report.push(json!({
            "body": b.name, "method": if m.hexes == 0 { "tet" } else { "recombined" },
            "hex8": m.hexes, "pyramid5": m.pyramids, "tet4": m.elements.len() - m.hexes - m.pyramids,
            "hex_volume_fraction": (m.hex_volume / volume).clamp(0.0, 1.0),
            "elements": m.elements.len(), "nodes": b.mesh.points.len(), "boundary_faces": m.boundary.len(),
            "volume": volume,
        }));
    }
    let (nodes, elements) = (model.nodes().len(), model.elements().len());
    s.model = model;
    Ok(cleared.report(s, json!({"bodies": report, "nodes": nodes, "elements": elements})))
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
    use fieldcraft_geometry::{SurfaceMesh, open_design, tagged_mesh::TaggedSurface};

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
    fn an_imported_tetrahedron_reports_zero_hexes_and_keeps_its_boundary() {
        let mesh = SurfaceMesh {
            positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            triangles: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
            tri_face: vec![0, 1, 2, 3],
            ..SurfaceMesh::default()
        };
        let mut s = Session::default();
        s.surfaces.push(Arc::new(TaggedSurface { name: "Tet".into(), mesh, faces: 4, original: None }));
        let r = s.execute("mesh.hex.dominant", &json!({"surface": 0, "size": 2000.0})).unwrap();
        let b = &r["bodies"][0];
        assert_eq!(b["method"], "tet");
        assert_eq!(b["hex_volume_fraction"], 0.0);
        assert_eq!(b["tet4"], 1);
        assert_eq!(b["boundary_faces"], 4);
        assert!(s.execute("mesh.hex.dominant", &json!({"surface": 0, "size": 2000.0, "max_tets": 1})).is_err());
        assert_eq!(s.model.elements().len(), 1);
    }
}
