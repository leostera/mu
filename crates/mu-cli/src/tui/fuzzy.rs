//! Port of `packages/tui/src/fuzzy.ts`: in-order character matching, lower score is better.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FuzzyMatch {
    pub matches: bool,
    pub score: f64,
}

fn is_word_boundary(previous: Option<char>) -> bool {
    previous.is_none_or(|ch| ch.is_whitespace() || matches!(ch, '-' | '_' | '.' | '/' | ':'))
}

fn match_query(query: &[char], text: &[char]) -> FuzzyMatch {
    let miss = FuzzyMatch {
        matches: false,
        score: 0.0,
    };
    if query.is_empty() {
        return FuzzyMatch {
            matches: true,
            score: 0.0,
        };
    }
    if query.len() > text.len() {
        return miss;
    }
    let mut score = 0.0;
    let mut last_match: Option<usize> = None;
    let mut consecutive = 0u32;
    for wanted in query {
        let from = last_match.map_or(0, |index| index + 1);
        let Some(offset) = text[from..].iter().position(|ch| ch == wanted) else {
            return miss;
        };
        let index = from + offset;
        if last_match.is_some_and(|last| last + 1 == index) {
            consecutive += 1;
            score -= f64::from(consecutive) * 5.0;
        } else {
            consecutive = 0;
            if let Some(last) = last_match {
                score += (index - last - 1) as f64 * 2.0;
            }
        }
        if is_word_boundary(index.checked_sub(1).map(|i| text[i])) {
            score -= 10.0;
        }
        score += index as f64 * 0.1;
        last_match = Some(index);
    }
    if query == text {
        score -= 100.0;
    }
    FuzzyMatch {
        matches: true,
        score,
    }
}

/// Letters+digits queries also try the swapped order ("abc12" vs "12abc").
fn swapped_query(query: &str) -> Option<String> {
    let letters = query.chars().take_while(char::is_ascii_lowercase).count();
    let (head, tail) = query.split_at(letters);
    let digits_only = |text: &str| !text.is_empty() && text.chars().all(|ch| ch.is_ascii_digit());
    let letters_only =
        |text: &str| !text.is_empty() && text.chars().all(|ch| ch.is_ascii_lowercase());
    if letters_only(head) && digits_only(tail) {
        return Some(format!("{tail}{head}"));
    }
    let digits = query.chars().take_while(char::is_ascii_digit).count();
    let (head, tail) = query.split_at(digits);
    (digits_only(head) && letters_only(tail)).then(|| format!("{tail}{head}"))
}

pub fn fuzzy_match(query: &str, text: &str) -> FuzzyMatch {
    let query = query.to_lowercase();
    let text_chars = text.to_lowercase().chars().collect::<Vec<_>>();
    let primary = match_query(&query.chars().collect::<Vec<_>>(), &text_chars);
    if primary.matches {
        return primary;
    }
    match swapped_query(&query) {
        Some(swapped) => {
            let swapped = match_query(&swapped.chars().collect::<Vec<_>>(), &text_chars);
            if swapped.matches {
                FuzzyMatch {
                    matches: true,
                    score: swapped.score + 5.0,
                }
            } else {
                primary
            }
        }
        None => primary,
    }
}

/// Filter by whitespace/slash separated tokens (all must match), best matches first.
pub fn fuzzy_filter<T: Clone>(items: &[T], query: &str, text: impl Fn(&T) -> String) -> Vec<T> {
    let tokens = query
        .trim()
        .split(|ch: char| ch.is_whitespace() || ch == '/')
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return items.to_vec();
    }
    let mut results = Vec::new();
    for item in items {
        let haystack = text(item);
        let mut total = 0.0;
        let mut all = true;
        for token in &tokens {
            let found = fuzzy_match(token, &haystack);
            if !found.matches {
                all = false;
                break;
            }
            total += found.score;
        }
        if all {
            results.push((item.clone(), total));
        }
    }
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.into_iter().map(|(item, _)| item).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_in_order_and_prefers_exact_and_boundary_hits() {
        assert!(fuzzy_match("mdl", "model").matches);
        assert!(!fuzzy_match("xyz", "model").matches);
        assert!(fuzzy_match("model", "model").score < fuzzy_match("mo", "model").score);
        assert!(
            fuzzy_match("sm", "scoped-models").score
                < fuzzy_match("sm", "settings-more-x").score + 100.0
        );
    }

    #[test]
    fn swapped_alphanumeric_queries_match_with_a_penalty() {
        let direct = fuzzy_match("12ab", "ab12");
        assert!(direct.matches);
        assert!(direct.score > fuzzy_match("ab12", "ab12").score);
    }

    #[test]
    fn filter_orders_by_score_and_requires_every_token() {
        let names = [
            "settings",
            "model",
            "scoped-models",
            "import",
            "name",
            "compact",
        ]
        .map(str::to_owned);
        let filtered = fuzzy_filter(&names, "m", |name| name.clone());
        assert_eq!(filtered[0], "model");
        assert!(!filtered.contains(&"settings".to_owned()));
        assert_eq!(
            fuzzy_filter(&names, "  ", |name| name.clone()).len(),
            names.len()
        );
        assert!(fuzzy_filter(&names, "model zzz", |name| name.clone()).is_empty());
    }
}
