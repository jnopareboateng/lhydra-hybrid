## Development Guidelines

### I. Core Principles & Philosophy

1.  **Simplicity First:** Prioritize clear, readable, and straightforward solutions.
2.  **DRY (Don't Repeat Yourself):** Maximize code reuse through functions, classes, and modules.
3.  **Clean Architecture:** Maintain a well-organized codebase.
    * **Strict File Size Limit:** Keep source code files under **400 lines**. Refactor aggressively if approaching or exceeding this limit by splitting into smaller, focused modules or helper files.
    * **Modularity:** Organize code into clearly separated modules, grouped logically (e.g., by feature or responsibility).

### II. Project Management & Context

4.  **Know the Plan:** Always read `PLANNING.md` at the start of a session to understand project architecture, goals, style, and constraints. Use established conventions.
5.  **Track Your Tasks:** Check `TASK.md` before starting work. Add new tasks with the date if not listed. Mark tasks as complete immediately upon finishing. Add any newly discovered sub-tasks or TODOs to `TASK.md`.
6.  **Understand Existing Code:** Before adding features, understand the current architecture and workflow.
7.  **Documentation Maintenance:** Update `README.md` for new features, dependency changes, or setup modifications.

### III. Development Workflow & Environment

8.  **Environment Awareness:** Write code considering all target environments (e.g., dev, test, prod).
9.  **Configuration:** Use specified configuration files (e.g., `training_config.yaml`). Never modify sensitive files (like `.env`) without explicit approval.
10. **Mocking:** Use mock data *only* for unit tests. Never use mock data in development or production builds.
11. **Environment Setup (Example):** Follow project-specific setup (e.g., Linux/WSL: `conda activate Lhydra`).

### IV. Code Quality & Best Practices (Python)

12. **Modern & Concise Python:** Use modern idioms, built-ins (e.g., comprehensions over loops where clear), and appropriate data structures to optimize for readability, performance, and conciseness. Apply functional patterns where suitable.
13. **Small, Focused Functions:** Prefer small functions with single responsibilities.
14. **Imports:** Use clear, consistent imports (prefer relative imports within packages).
15. **Error Handling:** Implement necessary error handling, but avoid overly verbose or redundant checks.
16. **Logging:** Keep logging professional, concise, and meaningful. Avoid excessive verbosity.

### V. Style, Conventions & Documentation

17. **Language & Formatting:** Use Python. Adhere strictly to PEP8 guidelines. Use `black` for automated formatting. Employ type hints for clarity and static analysis.
18. **Naming:** Use meaningful yet concise variable, function, and class names.
19. **Data Validation:** Use `pydantic` for data validation where applicable.
20. **Frameworks/Libraries:** Use specified project libraries (e.g., `FastAPI` for APIs, `SQLAlchemy`/`SQLModel` for ORM) if applicable.
21. **Docstrings:** Write Google-style docstrings for **every** function and class.
    ```python
    def example(param1: str) -> bool:
        """Brief summary of the function's purpose.

        Args:
            param1: Description of parameter 1.

        Returns:
            Description of the return value.
        """
        # ... function logic ...
        return True
    ```
22. **Comments:**
    * Strive for self-documenting code.
    * Comment non-obvious logic.
    * For complex algorithms or decisions, add an inline `# Reason:` comment explaining *why* the approach was chosen.

### VI. Testing & Reliability

23. **Mandatory Unit Tests:** Create Pytest unit tests for **all** new features (functions, classes, methods, routes). *(This overrides any previous guideline suggesting otherwise).*
24. **Test Coverage:** Include at least:
    * One test for the expected "happy path" usage.
    * At least one edge case test.
    * At least one failure case test (e.g., testing error handling).
25. **Test Location:** Place tests in a `/tests` directory mirroring the main application structure.
26. **Update Tests:** When updating logic, check and update corresponding unit tests accordingly.

### VII. AI Assistant Specific Rules

27. **Ask, Don't Assume:** If context is missing or requirements are unclear, ask for clarification.
28. **No Hallucination:** Only use known, verifiable Python libraries and functions. Confirm existence before use.
29. **Verify Paths:** Always confirm file paths and module names exist before referencing them.
30. **Respect Existing Code:** Never delete or overwrite code unless explicitly instructed or as part of an approved task from `TASK.md`.

### VIII. Pre-Submission Checklist

* [ ] **Functionality:** Does the code work correctly across all relevant environments (dev, test, prod)?
* [ ] **Optimality:** Could built-ins replace loops/logic? Are optimal data structures used? Is time/space complexity reasonable? Redundant operations eliminated?
* [ ] **Tests:** Are new features covered by unit tests (happy path, edge, failure)? Are existing tests updated if logic changed?
* [ ] **Style & Docs:** Is the code formatted (`black`)? Does it follow PEP8? Are type hints present? Do all functions/classes have docstrings? Is complex logic commented?
* [ ] **File Size:** Are all files below the 300-line limit?
* [ ] **Context Files:** Is `TASK.md` updated? Is `README.md` updated if necessary?
* [ ] **Configuration:** Have sensitive files (`.env`) been left unmodified (unless approved)?