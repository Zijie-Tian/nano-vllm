# Implementation Plan: COMPASS Metadata Offloading and TMAC Verification

## Phase 1: Metadata Buffer Allocation and Setup
- [x] Task: Implement `alloc_policy_metadata` in `COMPASSPolicy` to allocate pinned CPU memory for metadata.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Metadata Buffer Allocation and Setup' (Protocol in workflow.md)

## Phase 2: Metadata Offloading Implementation
- [x] Task: Implement Q and packed K-cache offloading in `COMPASSPolicy`.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Metadata Offloading Implementation' (Protocol in workflow.md)

## Phase 3: TMAC Accuracy Verification
- [x] Task: Develop a verification script (`tests/verify_tmac_offload_accuracy.py`).
- [x] Task: Conductor - User Manual Verification 'Phase 3: TMAC Accuracy Verification' (Protocol in workflow.md)
