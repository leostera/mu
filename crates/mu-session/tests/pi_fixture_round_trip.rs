use std::collections::HashSet;

use mu_session::{CURRENT_SESSION_VERSION, ParentReference, SessionDocument};

const PINNED_PI_FIXTURE: &str =
    include_str!("../../../reference/pi/packages/coding-agent/test/fixtures/large-session.jsonl");

#[test]
fn pinned_pi_fixture_survives_parse_and_write_without_semantic_changes() {
    let session = SessionDocument::parse_jsonl(PINNED_PI_FIXTURE)
        .expect("Pi's checked-in session fixture should parse");
    let serialized = session.to_jsonl().expect("write Pi-compatible JSONL");
    let reparsed = SessionDocument::parse_jsonl(&serialized)
        .expect("Mu's output should remain readable by Mu");

    assert_eq!(reparsed, session);
    assert!(session.header().is_some());
    assert_eq!(session.entries().count(), 1018);
}

#[test]
fn pinned_legacy_pi_fixture_migrates_to_v3_tree_entries() {
    let mut session = SessionDocument::parse_jsonl(PINNED_PI_FIXTURE)
        .expect("Pi's checked-in legacy session should parse");

    assert!(session.migrate_to_current_version());
    assert_eq!(
        session.header().and_then(|header| header.version),
        Some(CURRENT_SESSION_VERSION)
    );

    let entries = session.entries().collect::<Vec<_>>();
    let mut ids = HashSet::new();
    for entry in &entries {
        let id = entry.id.as_ref().expect("migrated entries have IDs");
        assert!(ids.insert(id.as_str()), "entry IDs must be unique");
        assert_ne!(entry.parent_id, ParentReference::LegacyMissing);
    }

    let migrated = session.to_jsonl().expect("serialize migrated session");
    assert_eq!(
        SessionDocument::parse_jsonl(&migrated).expect("parse migrated session"),
        session
    );
}
