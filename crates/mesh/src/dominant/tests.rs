use std::collections::BTreeMap;

use crate::quality::conformity::conformity;
use crate::quality::{GateOptions, validity_gate_elements};
use crate::tet::TaggedFace;

use super::*;

fn source(pyramid: bool, seam: bool) -> QualityMesh {
    let mut points = vec![[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]];
    let ring = [1, 2, 3, 7, 4, 5];
    let mut tets: Vec<_> = ring.iter().zip(ring.iter().cycle().skip(1)).map(|(&a, &b)| [0, 6, a, b]).collect();
    if pyramid {
        points.extend([[0.5, 0.5, 2.], [0., -1., 1.]]);
        tets.extend([[4, 5, 6, 8], [4, 6, 7, 8], [4, 5, 8, 9]]);
    }
    tet_mesh(points, tets, seam)
}

fn tet_mesh(points: Vec<[f64; 3]>, mut tets: Vec<[u32; 4]>, seam: bool) -> QualityMesh {
    let mut faces = BTreeMap::<_, Vec<_>>::new();
    for (i, tet) in tets.iter_mut().enumerate() {
        if element_quality(K::Tet4, &coordinates(&points, *tet).unwrap()).unwrap().validity() != Some(Validity::Valid) {
            tet.swap(0, 1);
        }
        for opposite in 0..4 {
            let tri = crate::tet::complex::face_toward(*tet, opposite);
            let face = if seam && sorted3(tri) == [0, 1, 2] { 1 } else { 0 };
            faces.entry(sorted3(tri)).or_default().push(TaggedFace { tet: i as u32, opposite: opposite as u8, face });
        }
    }
    QualityMesh {
        points,
        elements: tets.into_iter().map(|t| t.to_vec()).collect(),
        kind: K::Tet4,
        boundary: faces
            .into_values()
            .filter_map(|owners| match owners.as_slice() {
                [b] => Some(*b),
                _ => None,
            })
            .collect(),
        report: Default::default(),
    }
}

#[test]
fn joins_a_cube_to_the_remaining_tets_through_a_pyramid() {
    let source = source(true, false);
    let mesh = recombine(&source).unwrap();
    assert_eq!(mesh, recombine(&source).unwrap());
    let kinds: Vec<_> = mesh.elements.iter().map(|(k, _)| *k).collect();
    assert_eq!(kinds, [K::Hex8, K::Pyramid5, K::Tet4]);
    assert!((mesh.hex_volume - 1.0).abs() < 1e-12);

    let points: Vec<_> = source.points.iter().copied().enumerate().map(|(i, p)| (i as u32, p)).collect();
    let topology = conformity(&points, mesh.elements.iter().enumerate().map(|(i, (k, e))| (i as u32, *k, e.as_slice())), 0.0).unwrap();
    assert!(topology.is_conforming());
    let mut tagged: Vec<_> = mesh.boundary.iter().map(|&(e, f, _)| (e as u32, f)).collect();
    tagged.sort_unstable();
    assert_eq!(tagged, topology.free_faces);
    let gate = validity_gate_elements(
        mesh.elements.iter().enumerate().map(|(i, (k, e))| (i as u32, *k, e.iter().map(|&v| source.points[v as usize]).collect::<Vec<_>>())),
        &GateOptions { expected_volume: Some(1.5), ..GateOptions::default() },
    )
    .unwrap();
    assert!(gate.passed(), "{gate:?}");
}

#[test]
fn preserves_a_boundary_seam_between_differently_tagged_triangles() {
    let source = source(false, true);
    let mesh = recombine(&source).unwrap();
    assert_eq!(mesh.hexes, 0);
    assert_eq!(mesh.elements.len(), 6);
    assert_eq!(mesh.boundary.iter().filter(|&&(_, _, f)| f == 1).count(), 1);
}

#[test]
fn two_hex_candidates_can_share_a_quad() {
    let first = source(false, false);
    let mut points = first.points;
    points.extend([[0., 0., 2.], [1., 0., 2.], [1., 1., 2.], [0., 1., 2.]]);
    let mut tets: Vec<[u32; 4]> = first.elements.into_iter().map(|e| e.try_into().unwrap()).collect();
    tets.extend(tets.clone().into_iter().map(|e| e.map(|n| n + 4)));
    let mesh = recombine(&tet_mesh(points, tets, false)).unwrap();
    assert_eq!(mesh.hexes, 2);
    assert_eq!(mesh.elements.len(), 2);
    assert_eq!(mesh.boundary.len(), 10);
}
