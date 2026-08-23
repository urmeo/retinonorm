# Security Policy

CortexProbe is an offline Python library and a set of analysis scripts. It runs no server,
exposes no service, ships no secrets or API keys, and makes no network requests: stimuli are
generated in-repo and the model under test is loaded locally.

## Supported versions

Only the latest `main` is supported. Fixes land on `main`; there are no backports.

## Reporting a vulnerability

Please report privately rather than opening a public issue: use the repository's
**Security** tab, then **Report a vulnerability**, to open a private advisory.

Expect an acknowledgement within a few days. The realistic surface is the dependency chain
(NumPy, SciPy, and the optional PyTorch extra) and the handling of user-supplied arrays.
