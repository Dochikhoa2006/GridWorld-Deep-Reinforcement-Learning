# Security Policy

## Supported versions

This project does not currently publish versioned security-maintenance releases.
Security fixes are applied to the latest commit on the default branch. Historical
commits, local experiment branches, model checkpoints, and generated artifacts are
not supported.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Report it by email to **dochikhoa2006@gmail.com** with the subject
`[Gridworld-10 security]`. Include:

- the affected commit or version;
- the operating system, Python version, and execution method;
- a minimal reproduction or proof of concept;
- the likely impact;
- whether the issue is already public; and
- a safe way to contact you for follow-up.

Do not include credentials, private datasets, or unnecessary personal information.
Please avoid accessing data you do not own, disrupting third-party services, or
publishing exploit details before a fix is available.

The maintainer will make a best-effort acknowledgement within seven days, assess the
report, and provide status updates when practical. Timing for a fix depends on
severity, reproducibility, and maintainer availability; this is an open-source
project and no service-level guarantee is offered.

## Security considerations for users

- **Checkpoints:** PyTorch checkpoint formats may rely on Python serialization.
  Load only files from trusted sources. A model file is not safe merely because its
  extension is `.pt` or `.pth`.
- **Datasets:** Treat CSV files as untrusted input. Review their origin and size, and
  do not weaken schema validation to accommodate malformed downloads.
- **Repository history:** Ignore rules protect future commits, not prior blobs.
  Audit every reachable ref before asserting that third-party data is absent from a
  public repository, and coordinate any history rewrite as a destructive migration.
- **Dependencies:** Install from the pinned dependency file in an isolated
  environment and review automated dependency alerts before upgrading.
- **Containers:** The image uses a non-root user, but bind mounts still expose host
  files according to host/container permissions. Mount the dataset read-only.
- **Artifacts:** Configurations and logs may reveal local paths or environment
  details. Inspect them before publishing.
- **Secrets:** This project requires no API token for local training. Do not commit
  Kaggle credentials, cloud keys, CI secrets, or private dataset URLs.

Issues concerning the upstream dataset's availability, accuracy, or licensing are
not software vulnerabilities in this repository. Report those to the dataset owner.
