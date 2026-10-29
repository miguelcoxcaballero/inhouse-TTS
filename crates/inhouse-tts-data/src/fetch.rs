//! Resumable HTTP downloads and archive extraction.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

fn agent() -> Result<ureq::Agent, String> {
    let mut cfg = ureq::Agent::config_builder().timeout_global(None);
    // Honour a custom CA bundle (corporate or sandbox proxies).
    if let Some(path) = std::env::var_os("SSL_CERT_FILE") {
        let pem = fs::read(&path).map_err(|e| format!("{path:?}: {e}"))?;
        let certs: Vec<_> = ureq::tls::parse_pem(&pem)
            .filter_map(|i| match i {
                Ok(ureq::tls::PemItem::Certificate(c)) => Some(c),
                _ => None,
            })
            .collect();
        cfg = cfg.tls_config(
            ureq::tls::TlsConfig::builder()
                .root_certs(ureq::tls::RootCerts::new_with_certs(&certs))
                .build(),
        );
    }
    Ok(cfg.build().into())
}

/// Downloads `url` to `dest`, resuming a partial `dest.part` file.
pub fn download(url: &str, dest: &Path) -> Result<(), String> {
    if dest.exists() {
        return Ok(());
    }
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let part = dest.with_extension("part");
    let have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let agent = agent()?;
    let mut req = agent.get(url);
    if have > 0 {
        req = req.header("Range", &format!("bytes={have}-"));
    }
    let mut resp = req.call().map_err(|e| format!("{url}: {e}"))?;
    let resumed = resp.status() == 206;
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(|n| if resumed { n + have } else { n });
    let mut out = OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&part)
        .map_err(|e| e.to_string())?;
    let mut done = if resumed { have } else { 0 };
    let mut reader = resp.body_mut().as_reader();
    let mut buf = vec![0u8; 1 << 20];
    let t0 = Instant::now();
    let mut last = 0.0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{url}: {e}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        done += n as u64;
        let t = t0.elapsed().as_secs_f64();
        if t - last > 10.0 {
            last = t;
            match total {
                Some(tot) => eprintln!(
                    "  {}: {:.0}/{:.0} MB ({:.1} MB/s)",
                    dest.display(),
                    done as f64 / 1e6,
                    tot as f64 / 1e6,
                    done as f64 / 1e6 / t
                ),
                None => eprintln!("  {}: {:.0} MB", dest.display(), done as f64 / 1e6),
            }
        }
    }
    if let Some(tot) = total {
        if done != tot {
            return Err(format!(
                "{url}: incomplete download ({done} of {tot} bytes)"
            ));
        }
    }
    fs::rename(&part, dest).map_err(|e| e.to_string())
}

/// Extracts `.zip`, `.tar.gz`/`.tgz` or `.tar` archives into `dir`.
pub fn extract(archive: &Path, dir: &Path) -> Result<(), String> {
    let marker = dir.join(".extracted");
    if marker.exists() {
        return Ok(());
    }
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let name = archive.to_string_lossy().to_lowercase();
    let f = File::open(archive).map_err(|e| e.to_string())?;
    if name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(f).map_err(|e| e.to_string())?;
        z.extract(dir).map_err(|e| e.to_string())?;
    } else if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        tar::Archive::new(flate2::read::GzDecoder::new(f))
            .unpack(dir)
            .map_err(|e| e.to_string())?;
    } else if name.ends_with(".tar") {
        tar::Archive::new(f)
            .unpack(dir)
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!("unknown archive type: {name}"));
    }
    File::create(marker).map_err(|e| e.to_string())?;
    Ok(())
}
