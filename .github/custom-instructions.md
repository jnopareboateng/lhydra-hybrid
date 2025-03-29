# Lhydra Development Guidelines

## Core Principles
- **Simplicity First**: Prefer simple, readable solutions over complex ones
- **Don't Repeat Yourself (DRY)**: Reuse existing components when possible
- **Clean Architecture**: Keep files under 300 lines and the codebase organized refactor at this point if exceeding 300 lines

## Code Quality Standards

### Python Best Practices
- Write concise code with modern Python idioms and built-ins
- Use list/dict comprehensions over explicit loops
- Apply functional programming patterns where appropriate
- Choose appropriate data structures to minimize complexity
- Optimize for both readability and performance
- Avoid verbose logging keep it professional and concise

### Style Guidelines
- Use meaningful but concise variable names
- Minimize comments - code should be self-documenting
- Avoid unnecessary error handling
- Favor small, focused functions over large, complex ones

## Project-Specific Instructions

### Environment Setup
- Run commands in Linux environment:
  ```
  wsl
  conda activate Lhydra
  ```
- Configurations are in `training_config.yaml`
- Project scope is defined in `README.md`

### Development Workflow
- Understand the existing architecture before implementing new features
- Consider all environments: dev, test, and prod in your code
- Mock data only for tests, never for dev or prod
- Don't write test files for implementation go straight to the point
- Never modify `.env` files without explicit approval

### Before Code Submission
- Verify if built-ins could replace multiple lines
- Check if your solution uses optimal data structures
- Eliminate redundant operations
- Ensure optimal time and space complexity
- Confirm your implementation works across all environments