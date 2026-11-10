//! Hex-dominant recombination of a conforming linear tet mesh.
//!
//! Pellerin, Johnen, Verhetsel and Remacle, *Identifying combinations of tetrahedra into
//! hexahedra* (CAD 105, 2018, §§3.3 and 4.3): recover a candidate's two triangles per face,
//! then select non-overlapping, face-compatible cells. This first version searches the
//! six-tet fan around a body diagonal. It accepts only convex hexes with planar faces,
//! so replacing a fan preserves its exact polyhedral boundary. A quad meets an identical
//! hex quad, or a pyramid made from the two outside tets when they share an apex.
//! Every interior face is shared by two cells with the same face connectivity.

use std::collections::{BTreeMap, VecDeque};

use fieldcraft_model::{ElementKind as K, progress};

use crate::predicates::{Point3, Sign, tet_volume_sign};
use crate::quality::{Metric, Validity, element_quality};
use crate::tet::{
    QualityMesh,
    complex::{face_toward, sorted3},
};
use crate::{Error, Result};

type Tri = [u32; 3];
type Quad = [u32; 4];
type FaceIndex = [(Tri, u32)];

/// Caps the face index and the bounded candidate search.
pub const MAX_SOURCE_TETS: usize = 500_000;

#[derive(Debug, PartialEq)]
pub struct Recombined {
    pub elements: Vec<(K, Vec<u32>)>,
    /// (zero-based output element, one-based local face, source face tag).
    pub boundary: Vec<(usize, u8, u32)>,
    pub hexes: usize,
    pub pyramids: usize,
    pub hex_volume: f64,
}

#[derive(Clone, Copy)]
struct Face {
    quad: Quad,
    triangles: [Tri; 2],
    outside: [Option<u32>; 2],
}

struct Candidate {
    nodes: [u32; 8],
    tets: [u32; 6],
    faces: [Face; 6],
    quality: f64,
    volume: f64,
}

#[derive(Clone, Copy)]
enum Join {
    Boundary(u32),
    Hex(usize),
    Pyramid([u32; 5], [u32; 2]),
    Missing,
}

fn invalid() -> Error {
    Error::Invalid("recombination requires a conforming tet4 mesh with complete boundary tags".into())
}

fn at(points: &[[f64; 3]], node: u32) -> Result<[f64; 3]> {
    points.get(node as usize).copied().ok_or_else(invalid)
}

fn coordinates<const N: usize>(points: &[[f64; 3]], nodes: [u32; N]) -> Result<[[f64; 3]; N]> {
    let mut out = [[0.0; 3]; N];
    for (p, node) in out.iter_mut().zip(nodes) {
        *p = at(points, node)?;
    }
    Ok(out)
}

fn outside(index: &FaceIndex, tri: Tri, inside: impl Fn(u32) -> bool) -> Option<u32> {
    let start = index.partition_point(|&(key, _)| key < tri);
    index.get(start..)?.iter().take_while(|&&(key, _)| key == tri).find(|&&(_, t)| !inside(t)).map(|&(_, t)| t)
}

fn ring(tets: &[[u32; 4]], index: &FaceIndex, seed: u32, a: u32, b: u32) -> Option<([u32; 6], [u32; 6])> {
    let mut first = tets.get(seed as usize)?.iter().copied().filter(|&v| v != a && v != b);
    let mut u = first.next()?;
    let mut v = first.next()?;
    let mut around = [0; 6];
    let mut cells = [0; 6];
    let mut t = seed;
    for (i, (r, cell)) in around.iter_mut().zip(&mut cells).enumerate() {
        *r = u;
        *cell = t;
        let next = outside(index, sorted3([a, b, v]), |id| id == t)?;
        if i == 5 {
            if next != seed {
                return None;
            }
        } else if next <= seed {
            return None;
        }
        let w = tets.get(next as usize)?.iter().copied().find(|&x| x != a && x != b && x != v)?;
        u = v;
        v = w;
        t = next;
    }
    let mut unique = around;
    unique.sort_unstable();
    if unique.windows(2).any(|p| p.first() == p.get(1)) {
        return None;
    }
    cells.sort_unstable();
    Some((around, cells))
}

fn candidate(
    index: &FaceIndex,
    points: &[[f64; 3]],
    boundary: &BTreeMap<Tri, u32>,
    diagonal: [u32; 2],
    ring: [u32; 6],
    cells: [u32; 6],
) -> Result<Option<Candidate>> {
    let [a, g] = diagonal;
    for direction in [1, 5] {
        for start in 0..2 {
            // Even rotations name the same face pairing. The reverse covers the other handedness.
            let r: [u32; 6] = std::array::from_fn(|k| ring[(start + direction * k) % 6]);
            let [b, c, d, h, e, f] = r;
            let nodes = [a, b, c, d, e, f, g, h];
            let xyz = coordinates(points, nodes)?;
            let exact: Vec<_> = xyz.iter().filter_map(|&[x, y, z]| Point3::new(x, y, z).ok()).collect();
            let Ok(exact) = <[Point3; 8]>::try_from(exact) else { continue };
            let mut faces = Vec::with_capacity(6);
            for face in K::Hex8.faces() {
                let &[i, j, k, l] = *face else { return Err(invalid()) };
                let [Some(p), Some(q), Some(r), Some(s)] = [i, j, k, l].map(|n| exact.get(n)) else { return Err(invalid()) };
                if tet_volume_sign(p, q, r, s) != Sign::Zero
                    || exact.iter().enumerate().any(|(m, x)| !face.contains(&m) && tet_volume_sign(p, q, r, x) != Sign::Negative)
                {
                    break;
                }
                let [Some(&i), Some(&j), Some(&k), Some(&l)] = [i, j, k, l].map(|n| nodes.get(n)) else { return Err(invalid()) };
                let mut quad = [i, j, k, l];
                let Some(diagonal_end) = quad.iter().position(|v| diagonal.contains(v)) else { return Err(invalid()) };
                quad.rotate_left(diagonal_end);
                let [p, q, r, s] = quad;
                let mut triangles = [sorted3([p, q, r]), sorted3([p, r, s])];
                triangles.sort_unstable();
                let outside = triangles.map(|tri| outside(index, tri, |t| cells.contains(&t)));
                match outside {
                    [None, None] if boundary.get(&triangles[0]).is_some_and(|tag| Some(tag) == boundary.get(&triangles[1])) => {}
                    [Some(_), Some(_)] => {}
                    _ => break,
                }
                faces.push(Face { quad, triangles, outside });
            }
            if let Ok(faces) = <[Face; 6]>::try_from(faces) {
                let q = element_quality(K::Hex8, &xyz)?;
                if q.validity() == Some(Validity::Valid) {
                    return Ok(Some(Candidate { nodes, tets: cells, faces, quality: q.get(Metric::ScaledJacobian).unwrap_or(0.0), volume: q.size }));
                }
            }
        }
    }
    Ok(None)
}

fn boundary_tags(mesh: &QualityMesh, tets: &[[u32; 4]], index: &FaceIndex) -> Result<BTreeMap<Tri, u32>> {
    let mut tags = BTreeMap::new();
    for b in &mesh.boundary {
        let tet = tets.get(b.tet as usize).copied().ok_or_else(invalid)?;
        if b.opposite >= 4 {
            return Err(invalid());
        }
        let face = face_toward(tet, usize::from(b.opposite));
        if tags.insert(sorted3(face), b.face).is_some() {
            return Err(invalid());
        }
    }
    for run in index.chunk_by(|a, b| a.0 == b.0) {
        match run {
            [(tri, _)] if tags.contains_key(tri) => {}
            [(tri, _), _] if !tags.contains_key(tri) => {}
            _ => return Err(invalid()),
        }
    }
    Ok(tags)
}

fn join(
    face: &Face,
    candidates: &[Candidate],
    owner: &[Option<usize>],
    reserved: &mut [bool],
    tags: &BTreeMap<Tri, u32>,
    tets: &[[u32; 4]],
    points: &[[f64; 3]],
) -> Result<Join> {
    let [Some(a), Some(b)] = face.outside else {
        return tags.get(&face.triangles[0]).copied().map(Join::Boundary).ok_or_else(invalid);
    };
    let owners = [a, b].map(|i| owner.get(i as usize).copied().flatten());
    match owners {
        [Some(i), Some(j)] if i == j && candidates.get(i).is_some_and(|c| c.faces.iter().any(|f| f.triangles == face.triangles)) => {
            return Ok(Join::Hex(i));
        }
        [None, None] if a != b => {}
        _ => return Ok(Join::Missing),
    }
    if [a, b].iter().any(|&i| reserved.get(i as usize) != Some(&false)) {
        return Ok(Join::Missing);
    }
    let apex = |t: u32| tets.get(t as usize)?.iter().copied().find(|v| !face.quad.contains(v));
    let Some(v) = apex(a).filter(|&v| Some(v) == apex(b)) else { return Ok(Join::Missing) };
    let [i, j, k, l] = face.quad;
    let pyramid = [i, j, k, l, v];
    if element_quality(K::Pyramid5, &coordinates(points, pyramid)?)?.validity() != Some(Validity::Valid) {
        return Ok(Join::Missing);
    }
    for t in [a, b] {
        *reserved.get_mut(t as usize).ok_or_else(invalid)? = true;
    }
    Ok(Join::Pyramid(pyramid, [a, b]))
}

/// Recombine the output of the linear tet mesher, preserving its points, boundary and face tags.
pub fn recombine(mesh: &QualityMesh) -> Result<Recombined> {
    if mesh.kind != K::Tet4 {
        return Err(invalid());
    }
    if mesh.elements.len() > MAX_SOURCE_TETS {
        return Err(Error::Limit(format!("hex recombination is limited to {MAX_SOURCE_TETS} source tetrahedra")));
    }
    let mut tets = Vec::with_capacity(mesh.elements.len());
    let mut index = Vec::with_capacity(mesh.elements.len() * 4);
    for (i, cell) in mesh.elements.iter().enumerate() {
        if i.is_multiple_of(1024) {
            progress::check()?;
        }
        let tet: [u32; 4] = cell.as_slice().try_into().map_err(|_| invalid())?;
        let mut nodes = tet;
        nodes.sort_unstable();
        if nodes.windows(2).any(|pair| pair.first() == pair.get(1)) || nodes.iter().any(|&node| mesh.points.get(node as usize).is_none()) {
            return Err(invalid());
        }
        let [a, b, c, d] = tet;
        for tri in [[a, b, c], [a, b, d], [a, c, d], [b, c, d]] {
            index.push((sorted3(tri), i as u32));
        }
        tets.push(tet);
    }
    index.sort_unstable();
    let mut tags = boundary_tags(mesh, &tets, &index)?;
    let mut candidates = Vec::new();
    for (i, &[a, b, c, d]) in tets.iter().enumerate() {
        if i.is_multiple_of(1024) {
            progress::check()?;
        }
        for [a, b] in [[a, b], [a, c], [a, d], [b, c], [b, d], [c, d]] {
            if let Some((r, cells)) = ring(&tets, &index, i as u32, a, b)
                && let Some(candidate) = candidate(&index, &mesh.points, &tags, [a, b], r, cells)?
            {
                candidates.push(candidate);
            }
        }
    }
    drop(index);
    candidates.sort_unstable_by(|a, b| b.quality.total_cmp(&a.quality).then_with(|| a.nodes.cmp(&b.nodes)));

    // Greedily reserve non-overlapping hexes, then prune any whose quad cannot be matched.
    let mut owner = vec![None; tets.len()];
    let mut chosen = vec![false; candidates.len()];
    for (i, c) in candidates.iter().enumerate() {
        if c.tets.iter().all(|&t| owner.get(t as usize) == Some(&None)) {
            for t in c.tets {
                *owner.get_mut(t as usize).ok_or_else(invalid)? = Some(i);
            }
            if let Some(flag) = chosen.get_mut(i) {
                *flag = true;
            }
        }
    }
    let mut reserved = vec![false; tets.len()];
    let mut joins = vec![[Join::Missing; 6]; candidates.len()];
    let mut reject = VecDeque::new();
    for (i, ((c, &keep), joins)) in candidates.iter().zip(&chosen).zip(&mut joins).enumerate() {
        if keep {
            for (face, slot) in c.faces.iter().zip(joins) {
                *slot = join(face, &candidates, &owner, &mut reserved, &tags, &tets, &mesh.points)?;
                if matches!(slot, Join::Missing) {
                    reject.push_back(i);
                }
            }
        }
    }
    while let Some(i) = reject.pop_front() {
        let keep = chosen.get_mut(i).ok_or_else(invalid)?;
        if std::mem::replace(keep, false) {
            for link in joins.get(i).ok_or_else(invalid)? {
                if let Join::Hex(other) = *link {
                    reject.push_back(other);
                }
            }
        }
    }

    let mut result = Recombined { elements: Vec::new(), boundary: Vec::new(), hexes: 0, pyramids: 0, hex_volume: 0.0 };
    let mut consumed = vec![false; tets.len()];
    for ((c, keep), joins) in candidates.into_iter().zip(chosen).zip(joins) {
        if !keep {
            continue;
        }
        let element = result.elements.len();
        result.elements.push((K::Hex8, c.nodes.to_vec()));
        result.hexes += 1;
        result.hex_volume += c.volume;
        for t in c.tets {
            *consumed.get_mut(t as usize).ok_or_else(invalid)? = true;
        }
        for (k, (face, link)) in c.faces.into_iter().zip(joins).enumerate() {
            match link {
                Join::Boundary(tag) => {
                    for tri in face.triangles {
                        tags.remove(&tri).ok_or_else(invalid)?;
                    }
                    result.boundary.push((element, (k + 1) as u8, tag));
                }
                Join::Pyramid(nodes, ts) => {
                    for t in ts {
                        *consumed.get_mut(t as usize).ok_or_else(invalid)? = true;
                    }
                    result.elements.push((K::Pyramid5, nodes.to_vec()));
                    result.pyramids += 1;
                }
                Join::Hex(_) => {}
                Join::Missing => return Err(invalid()),
            }
        }
    }
    result.elements.extend(tets.into_iter().zip(consumed).filter(|&(_, taken)| !taken).map(|(t, _)| (K::Tet4, t.to_vec())));
    for (i, (kind, nodes)) in result.elements.iter().enumerate() {
        if *kind == K::Hex8 {
            continue;
        }
        for (k, face) in kind.faces().iter().enumerate() {
            if let &[a, b, c] = *face {
                let [Some(&a), Some(&b), Some(&c)] = [a, b, c].map(|j| nodes.get(j)) else { return Err(invalid()) };
                if let Some(tag) = tags.remove(&sorted3([a, b, c])) {
                    result.boundary.push((i, (k + 1) as u8, tag));
                }
            }
        }
    }
    if !tags.is_empty() {
        return Err(invalid());
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
