//! Credential storage in `auth.json` (Pi's `AuthStorage`): one type-tagged credential per provider.
//!
//! Config-value templates (`$VAR`, `!command`) are not resolved; keys are used as stored.

use std::{collections::BTreeMap, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    ApiKey {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        key: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        env: Option<BTreeMap<String, String>>,
    },
    Oauth {
        access: String,
        refresh: String,
        expires: f64,
        #[serde(flatten)]
        extra: BTreeMap<String, Value>,
    },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AuthError {
    #[error("Failed to read auth.json: {0}")]
    Read(String),
    #[error("Invalid auth.json: expected an object")]
    NotAnObject,
    #[error("Invalid auth.json credential for provider \"{0}\"")]
    InvalidCredential(String),
    #[error("Failed to write auth.json: {0}")]
    Write(String),
}

/// All stored credentials. A missing file is an empty store.
pub fn load(path: &Path) -> Result<BTreeMap<String, Credential>, AuthError> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(error) => return Err(AuthError::Read(error.to_string())),
    };
    let value: Value = serde_json::from_str(text.strip_prefix('\u{feff}').unwrap_or(&text))
        .map_err(|error| AuthError::Read(error.to_string()))?;
    let Value::Object(entries) = value else {
        return Err(AuthError::NotAnObject);
    };
    entries
        .into_iter()
        .map(|(provider, credential)| {
            serde_json::from_value::<Credential>(credential)
                .map(|credential| (provider.clone(), credential))
                .map_err(|_| AuthError::InvalidCredential(provider))
        })
        .collect()
}

fn save(path: &Path, credentials: &BTreeMap<String, Credential>) -> Result<(), AuthError> {
    let write = || -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_string_pretty(credentials)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(temporary, path)
    };
    write().map_err(|error| AuthError::Write(error.to_string()))
}

pub fn set_api_key(path: &Path, provider: &str, key: &str) -> Result<(), AuthError> {
    let mut credentials = load(path)?;
    credentials.insert(
        provider.to_owned(),
        Credential::ApiKey {
            key: Some(key.to_owned()),
            env: None,
        },
    );
    save(path, &credentials)
}

/// Store any credential (for example the result of an OAuth login) for `provider`.
pub fn set_credential(
    path: &Path,
    provider: &str,
    credential: Credential,
) -> Result<(), AuthError> {
    let mut credentials = load(path)?;
    credentials.insert(provider.to_owned(), credential);
    save(path, &credentials)
}

/// Remove a provider's credential; returns whether one existed.
pub fn remove(path: &Path, provider: &str) -> Result<bool, AuthError> {
    let mut credentials = load(path)?;
    let existed = credentials.remove(provider).is_some();
    if existed {
        save(path, &credentials)?;
    }
    Ok(existed)
}

/// The stored API key for `provider`, if it has an `api_key` credential with a key.
pub fn api_key(path: &Path, provider: &str) -> Option<String> {
    match load(path).ok()?.remove(provider)? {
        Credential::ApiKey { key, .. } => key.filter(|key| !key.is_empty()),
        Credential::Oauth { .. } => None,
    }
}

/// Where a provider's credential comes from, in Pi's priority order (after `--api-key`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialSource {
    /// A stored API key in `auth.json`.
    AuthJson(String),
    /// A stored OAuth credential in `auth.json` (no API key available).
    OAuth,
    /// The provider's `apiKey` in `models.json`, resolved.
    ModelsJson(String),
    None,
}

pub fn credential_source(agent_dir: &Path, provider: &str) -> CredentialSource {
    match load(&agent_dir.join("auth.json"))
        .ok()
        .and_then(|mut c| c.remove(provider))
    {
        Some(Credential::ApiKey { key: Some(key), .. }) if !key.is_empty() => {
            return CredentialSource::AuthJson(key);
        }
        Some(Credential::Oauth { .. }) => return CredentialSource::OAuth,
        _ => {}
    }
    let configured = std::fs::read_to_string(agent_dir.join("models.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|models| {
            models["providers"][provider]["apiKey"]
                .as_str()
                .and_then(|key| crate::config_value::resolve(key, &Default::default()))
        });
    configured.map_or(CredentialSource::None, CredentialSource::ModelsJson)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mu-auth-{}/auth.json", uuid::Uuid::new_v4()))
    }

    #[test]
    fn stores_keys_atomically_with_private_permissions_and_removes_them() {
        let path = temp();
        assert!(
            load(&path).unwrap().is_empty(),
            "missing file is an empty store"
        );
        set_api_key(&path, "acme", "sk-1").unwrap();
        set_api_key(&path, "other", "sk-2").unwrap();
        assert_eq!(api_key(&path, "acme").as_deref(), Some("sk-1"));
        let saved: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            saved["acme"],
            serde_json::json!({"type":"api_key","key":"sk-1"})
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        assert!(remove(&path, "acme").unwrap());
        assert!(!remove(&path, "acme").unwrap());
        assert_eq!(api_key(&path, "acme"), None);
        assert_eq!(api_key(&path, "other").as_deref(), Some("sk-2"));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn validates_credentials_and_keeps_oauth_extras() {
        let path = temp();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            r#"{"a":{"type":"oauth","access":"x","refresh":"y","expires":1.5,"accountId":"acc"}}"#,
        )
        .unwrap();
        let loaded = load(&path).unwrap();
        assert!(
            matches!(&loaded["a"], Credential::Oauth { extra, .. } if extra["accountId"] == "acc")
        );
        assert_eq!(api_key(&path, "a"), None);
        std::fs::write(&path, r#"{"a":{"type":"oauth","access":"x"}}"#).unwrap();
        assert_eq!(
            load(&path).unwrap_err(),
            AuthError::InvalidCredential("a".into())
        );
        std::fs::write(&path, "[]").unwrap();
        assert_eq!(load(&path).unwrap_err(), AuthError::NotAnObject);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn credential_source_follows_auth_json_then_models_json() {
        let dir = temp().parent().unwrap().to_owned();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"a":{"apiKey":"from-models"},"b":{"apiKey":"$MU_SURELY_UNSET_VAR_2"}}}"#,
        )
        .unwrap();
        set_api_key(&dir.join("auth.json"), "a", "from-auth").unwrap();
        assert_eq!(
            credential_source(&dir, "a"),
            CredentialSource::AuthJson("from-auth".into())
        );
        assert_eq!(credential_source(&dir, "b"), CredentialSource::None);
        remove(&dir.join("auth.json"), "a").unwrap();
        assert_eq!(
            credential_source(&dir, "a"),
            CredentialSource::ModelsJson("from-models".into())
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
