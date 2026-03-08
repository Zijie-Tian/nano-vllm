# Product Guidelines: Nano-vLLM

## 1. Documentation & Prose Style
*   **Tone:** Educational and detailed, catering to ML engineers and researchers. Explain the "why" behind low-level optimizations while maintaining technical precision.
*   **Terminology:** Use a mix of standard LLM academic terms (e.g., Softmax, LSE, Q/K/V) and project-specific terminology (e.g., T-MAC, BLASST, sgDMA). Ensure consistency in how these terms are used across the codebase and documentation.
*   **Structure:** Break down complex topics into smaller, digestible sections with clear headings and logical flow.

## 2. Code Documentation Standards
*   **Architectural File Headers:** Every significant file (e.g., core engine modules, custom kernels) MUST have a high-level architectural summary at the top, explaining its purpose and how it fits into the broader system.
*   **Comprehensive Docstrings:** Every function and class should have a detailed docstring, clearly defining parameters, return types, and expected tensor shapes.
*   **Targeted In-line Comments:** Use in-line comments sparingly but effectively for critical logic blocks, complex mathematical operations, or hardware-specific optimizations. Avoid clutter; focus on "key points" and non-obvious logic.

## 3. Communication & Architecture Principles
*   **Modularity:** Maintain the lightweight nature of the project by keeping modules focused and independent.
*   **Resource Awareness:** All documentation and code should be mindful of GPU/CPU memory constraints and the specialized offloading pipeline.
*   **Clarity over Complexity:** Prioritize readable, idiomatic Python code unless a performance-critical kernel (Triton/CUDA) is necessary.
