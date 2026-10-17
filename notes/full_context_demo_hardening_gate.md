# Phase 6H — Local analyst demo disclosure + browser security gate

This branch preserves all existing frozen model artifacts, policy thresholds,
21-feature schema, SQLite journal and BLOCK-disabled behavior. No real payment
use, public exposure or validated fraud-detection claims are authorized.

## Changes under test
- Separate dashboard JavaScript/CSS served through Flask's authenticated
  static endpoint (replaces CSP-incompatible inline scripts/styles).
- Explicit always-visible limitation: exploratory policy selected only
  97 of 412 known PaySim policy frauds; 0 of 206 CASH_OUT frauds.
- Evidence cards use DOM textContent, not HTML injection. Full recorded
  JSON remains inspectable behind a disclosure control.
- Browser refresh and unhealthy state are explicit. Failed refresh clears
  potentially stale counts and rows. Browser-measured time includes UI refresh
  and is not server/model inference latency.
- Per-response CSP, anti-frame, MIME-sniffing and referrer restrictions.
- POST denies any supplied cross-origin Origin and bounds JSON body size.
  Basic authentication and no-store remain mandatory for localhost lab.

## Local acceptance
1. Fetch branch and run all tests.
2. Confirm page, CSS and JS load authenticated, with no CSP errors.
3. Confirm unauthenticated requests return 401, cross-origin POST returns
   403, and normal same-origin POST still returns 201.
4. Use disposable new SQLite journal. Submit two sequential synthetic
   TRANSFERs. View observed evidence, duplicate a request, stop and
   restart Flask with SAME privacy key and database path.
5. Confirm restored records and evidence after restart. Tamper and unhealthy
   tests must fail closed without a new decision.
6. Never bind Flask development server beyond 127.0.0.1; HTTP Basic over
   plain HTTP is acceptable only for the loopback simulation. A LinuxONE
   deployment requires TLS, additional origin/host, network, secrets,
   authorization, logging and failure-mode reviews before exposure.

## Limitations
- This is a presentation and narrow browser/API security improvement, not
  completion of the LinuxONE security gate.
- Existing PaySim CASH_OUT score/ranking weakness is unresolved.
- The API is not safe to expose publicly. Do not use for real payments.
