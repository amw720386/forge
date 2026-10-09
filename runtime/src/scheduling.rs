//! Pure token budgeting shared by GPU scheduling and CPU fairness tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    DecodeFirst,
    Fixed,
}
impl Policy {
    pub fn from_name(name: &str) -> Self {
        match name {
            "fixed" => Self::Fixed,
            _ => Self::DecodeFirst,
        }
    }
}
#[derive(Default)]
pub struct Planner {
    pub prefill_cursor: usize,
    decode_cursor: usize,
    prefill_turn: bool,
}
pub struct Plan {
    pub decode: Vec<usize>,
    pub prefill: Vec<(usize, usize)>,
}
impl Planner {
    /// Prefill entries are (slot, remaining tokens).
    /// Legacy behavior remains exact. Fixed enforces the total budget,
    /// reserve progress for prefill, and rotate decode when the budget is tiny.
    pub fn plan(
        &mut self,
        policy: Policy,
        budget: usize,
        chunk: usize,
        decode: &[usize],
        prefill: &[(usize, usize)],
        slots: usize,
    ) -> Plan {
        let budget = budget.max(1);
        let mut result = Plan {
            decode: Vec::new(),
            prefill: Vec::new(),
        };
        let decode_count = if policy == Policy::DecodeFirst {
            decode.len()
        } else if budget == 1 && !decode.is_empty() && !prefill.is_empty() {
            self.prefill_turn = !self.prefill_turn;
            if self.prefill_turn {
                0
            } else {
                1
            }
        } else {
            decode
                .len()
                .min(budget.saturating_sub(usize::from(!prefill.is_empty())))
        };
        for n in 0..decode_count {
            let i = if policy == Policy::DecodeFirst {
                n
            } else {
                (self.decode_cursor + n) % decode.len()
            };
            result.decode.push(decode[i]);
        }
        if !decode.is_empty() {
            self.decode_cursor = (self.decode_cursor + decode_count) % decode.len();
        }
        let mut left = budget.saturating_sub(decode_count);
        for distance in 0..slots {
            if left == 0 {
                break;
            }
            let slot = (self.prefill_cursor + distance) % slots;
            if let Some(&(_, remaining)) = prefill.iter().find(|&&(i, _)| i == slot) {
                let n = remaining.min(chunk.max(1)).min(left);
                if n > 0 {
                    result.prefill.push((slot, n));
                    left -= n;
                }
            }
        }
        if let Some(&(slot, _)) = result.prefill.last() {
            self.prefill_cursor = (slot + 1) % slots;
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_budgets_cannot_starve_either_side() {
        for policy in [Policy::Fixed] {
            for budget in [1, 2, 4, 8, 16] {
                let mut p = Planner::default();
                let mut decoded = [0usize; 8];
                let mut filled = [0usize; 8];
                for _ in 0..64 {
                    let plan = p.plan(
                        policy,
                        budget,
                        128,
                        &(0..8).collect::<Vec<_>>(),
                        &(8..16).map(|i| (i, 4096)).collect::<Vec<_>>(),
                        16,
                    );
                    assert!(
                        plan.decode.len() + plan.prefill.iter().map(|x| x.1).sum::<usize>()
                            <= budget
                    );
                    for i in plan.decode {
                        decoded[i] += 1;
                    }
                    for (i, n) in plan.prefill {
                        filled[i - 8] += n;
                    }
                }
                assert!(decoded.iter().all(|&n| n > 0));
                assert!(filled.iter().all(|&n| n > 0));
            }
        }
    }
    #[test]
    fn legacy_plan_and_ragged_cursor() {
        let mut p = Planner::default();
        let plan = p.plan(
            Policy::DecodeFirst,
            256,
            128,
            &[0, 1],
            &[(2, 10), (3, 300), (4, 300)],
            5,
        );
        assert_eq!(plan.decode, vec![0, 1]);
        assert_eq!(plan.prefill, vec![(2, 10), (3, 128), (4, 116)]);
        assert_eq!(p.prefill_cursor, 0);
    }
}
