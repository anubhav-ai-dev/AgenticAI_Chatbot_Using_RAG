# Security issue (separate from the redesign): missing ownership checks

Status: **recorded only. Not fixed, not in redesign scope.** The redesign is frontend-only and must not change backend security behavior.

Source: `docs/redesign/audit.md` (section 8/9, security observations).

## Finding
`GET /sessions/{id}`, `PATCH /sessions/{id}`, `DELETE /sessions/{id}` and message feedback take no `user_id` ownership check. Anyone who knows or guesses a session id can read, rename, or delete that session, or submit feedback on its messages. `user_id` is a client-supplied identifier and is not authentication.

## Recommended follow-up (owner decision, separate change)
- Verify session/message ownership server-side on every session and feedback route.
- Introduce real authentication rather than relying on a client-provided `user_id`.
- Add tests asserting cross-user access is rejected.

## Redesign constraints
- Frontend continues to send `user_id` exactly as today.
- No session data is exposed or logged in docs. Note: `docs/redesign/baseline/api-shapes.txt` contains real session titles; delete or gitignore it before committing.
