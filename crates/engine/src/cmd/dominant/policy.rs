//! Acceptance of unpublished candidates, shared by the structured, octree and mixed routes.

use fieldcraft_mesh::quality::{Metric, Validity, element_quality};
use fieldcraft_model::{ElementKind, progress};
use serde_json::{Value, json};

use crate::schema::ProblemKind;
use crate::{EngineError, Result};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Partition {
    Standard,
    Alternate,
}

impl Partition {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Alternate => "alternate",
        }
    }
}

pub(super) struct Policy {
    pub(super) min_sj: f64,
    pub(super) max_ratio: f64,
    pub(super) min_hex_fraction: f64,
    pub(super) partitions: &'static [Partition],
}

impl Policy {
    pub(super) fn parse(p: &Value) -> Result<Option<Self>> {
        let Some(value) = p.get("requirements") else { return Ok(None) };
        let obj = value.as_object().ok_or_else(|| EngineError::invalid("requirements", "expected an object"))?;
        let read = |key: &str, default: f64, min: f64, max: f64| match obj.get(key) {
            None => Ok(default),
            Some(v) => v
                .as_f64()
                .filter(|x| (min..=max).contains(x) && x.is_finite())
                .ok_or_else(|| EngineError::param(format!("requirements/{key}"), ProblemKind::NotInRange { min, max })),
        };
        for key in obj.keys() {
            if !matches!(key.as_str(), "min_scaled_jacobian" | "max_edge_ratio" | "min_hex_volume_fraction" | "partition") {
                return Err(EngineError::invalid("requirements", format!("unknown field `{}`", key.chars().take(64).collect::<String>())));
            }
        }
        let partitions: &[_] = match obj.get("partition").map(Value::as_str) {
            None | Some(Some("auto")) => &[Partition::Standard, Partition::Alternate],
            Some(Some("standard")) => &[Partition::Standard],
            Some(Some("alternate")) => &[Partition::Alternate],
            _ => return Err(EngineError::param("requirements/partition", ProblemKind::NotAChoice)),
        };
        Ok(Some(Self {
            min_sj: read("min_scaled_jacobian", super::LOW_SCALED_JACOBIAN, 0.0, 1.0)?,
            max_ratio: read("max_edge_ratio", 100.0, 1.0, f64::MAX)?,
            min_hex_fraction: read("min_hex_volume_fraction", 0.5, 0.0, 1.0)?,
            partitions,
        }))
    }

    pub(super) fn limits(&self) -> Value {
        json!({"min_scaled_jacobian":self.min_sj,"max_edge_ratio":self.max_ratio,"min_hex_volume_fraction":self.min_hex_fraction})
    }

    pub(super) fn fraction_violation(&self, hex_fraction: f64, all_hex: bool) -> Option<Value> {
        (hex_fraction < self.min_hex_fraction || (self.min_hex_fraction == 1.0 && !all_hex))
            .then(|| json!({"code":"min_hex_volume_fraction","actual":hex_fraction,"limit":self.min_hex_fraction,"all_hex":all_hex}))
    }

    pub(super) fn inspect<'a>(&self, nodes: &[[f64; 3]], elements: impl Iterator<Item = (ElementKind, &'a [u32])>) -> Result<Inspection> {
        let (mut min, mut max, mut invalid) = (None::<Measure>, None::<Measure>, None);
        for (i, (kind, ids)) in elements.enumerate() {
            if i.is_multiple_of(1024) {
                progress::check()?;
            }
            let mut points = [[0.0; 3]; 8];
            let points = points.get_mut(..kind.node_count()).ok_or_else(|| EngineError::Params("expected linear solid elements".into()))?;
            if ids.len() != points.len() {
                return Err(EngineError::Params("invalid candidate connectivity".into()));
            }
            let mut at = [0.0; 3];
            for (point, &id) in points.iter_mut().zip(ids) {
                *point = *nodes.get(id as usize).ok_or_else(|| EngineError::Params("invalid candidate node".into()))?;
                for (sum, coord) in at.iter_mut().zip(*point) {
                    *sum += coord;
                }
            }
            let at = at.map(|x| x / points.len() as f64);
            let q = element_quality(kind, points)?;
            let (Some(sj), Some(ratio)) = (q.get(Metric::ScaledJacobian), q.get(Metric::EdgeRatio)) else {
                invalid = Some(json!({"code":"invalid_element","candidate_element":i,"at":at}));
                break;
            };
            if q.validity() != Some(Validity::Valid) || !sj.is_finite() || !ratio.is_finite() {
                invalid = Some(json!({"code":"invalid_element","candidate_element":i,"at":at}));
                break;
            }
            if min.as_ref().is_none_or(|m| sj < m.value) {
                min = Some(Measure { value: sj, element: i, at });
            }
            if max.as_ref().is_none_or(|m| ratio > m.value) {
                max = Some(Measure { value: ratio, element: i, at });
            }
        }
        let mut violations = Vec::new();
        if let Some(invalid) = invalid {
            violations.push(invalid);
        } else if min.is_none() {
            violations.push(json!({"code":"empty_mesh"}));
        }
        if let Some(min) = &min
            && min.value < self.min_sj
        {
            violations.push(min.violation("min_scaled_jacobian", self.min_sj));
        }
        if let Some(max) = &max
            && max.value > self.max_ratio
        {
            violations.push(max.violation("max_edge_ratio", self.max_ratio));
        }
        Ok(Inspection {
            quality: json!({
                "min_scaled_jacobian":min.as_ref().map(|m| m.value),
                "low_scaled_jacobian":min.is_some_and(|m| m.value < super::LOW_SCALED_JACOBIAN),
                "max_edge_ratio":max.map(|m| m.value),
            }),
            violations,
        })
    }
}

struct Measure {
    value: f64,
    element: usize,
    at: [f64; 3],
}

impl Measure {
    fn violation(&self, code: &str, limit: f64) -> Value {
        json!({"code":code,"actual":self.value,"limit":limit,"candidate_element":self.element,"at":self.at})
    }
}

pub(super) struct Inspection {
    pub(super) quality: Value,
    pub(super) violations: Vec<Value>,
}
