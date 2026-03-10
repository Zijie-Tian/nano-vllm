# Implementation Plan: COMPASS Metadata Offloading and TMAC Verification

## Phase 1: Metadata Buffer Allocation and Setup
- [ ] Task: Implement `alloc_policy_metadata` in `COMPASSPolicy` to allocate pinned CPU memory for metadata.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Metadata Buffer Allocation and Setup' (Protocol in workflow.md)

## Phase 2: Metadata Offloading Implementation
- [ ] Task: Implement Q and packed K-cache offloading in `COMPASSPolicy`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Metadata Offloading Implementation' (Protocol in workflow.md)

## Phase 3: TMAC Accuracy Verification
- [ ] Task: Develop a verification script (`tests/verify_tmac_offload_accuracy.py`).
- [ ] Task: Conductor - User Manual Verification 'Phase 3: TMAC Accuracy Verification' (Protocol in workflow.md)
