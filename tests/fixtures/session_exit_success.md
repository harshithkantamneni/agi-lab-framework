```json
{
  "schema_version": "1.0",
  "session_id": "D-T301",
  "claimed_item_id": "wq-test01",
  "status": "success",
  "reason": "GRACEFUL_CHECKPOINT",
  "log_entry_text": "D-T301 (test, ~00:00 UTC): **Test success entry.** Fixture for post_director.py success-branch test.\n\nSTATUS=success / KEY_FINDING=test / FILES_MODIFIED=test / SUMMARY=test fixture content.\n",
  "current_md_patches": [
    {"old": "Status: P9 ANALYZE", "new": "Status: P10 OPEN"}
  ],
  "deliverables": [
    "tests/fixtures/dummy_deliverable.md"
  ],
  "next_action": {
    "type": "phase_advance",
    "priority": "normal",
    "program": "test_program",
    "created_by": "director_session_d_t301",
    "payload": {
      "from_phase": "P10",
      "to_phase": "P10_close",
      "context": "Test follow-on enqueue."
    }
  },
  "notes": "Test fixture for success branch."
}
```

reason: GRACEFUL_CHECKPOINT
session_id: D-T301

Test fixture body — runner's existing `reason:` parser still reads this line.
