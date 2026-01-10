# AI Tools Usage Declaration

**Project:** Machine Learning for Supply Chain Intelligence  
**Author:** Luca Gozzi  
**Course:** Data Science and Advanced Programming  
**Date:** January 2026

---

## Overview

This document provides transparency about how AI tools were used throughout this project, in accordance with the course's AI policy. AI tools served as **collaborators and learning accelerators**, not as replacements for understanding or critical thinking. Every design decision, line of code, and written section was reviewed, understood, and validated by the author.

---

## Phase 1: Ideation and Brainstorming

### Tools Used
- **ChatGPT (OpenAI)**
- **Perplexity AI**

### How They Were Used

During the initial project planning phase, I used ChatGPT and Perplexity to:

1. **Explore project ideas**: Brainstormed various supply chain analytics approaches and evaluated their feasibility within course constraints.

2. **Compare ML options**: Discussed trade-offs between different machine learning approaches (classification vs. regression vs. time-series forecasting) for supply chain problems.

3. **Research datasets**: Identified publicly available supply chain datasets on Kaggle and evaluated their suitability for the project scope.

4. **Understand domain concepts**: Clarified supply chain terminology and KPIs (e.g., OTDR, ABC analysis, vendor scoring) to ensure the project addressed real business problems.

### What I Learned
These initial conversations helped me narrow down from a broad idea ("supply chain analytics") to a specific, achievable goal ("delivery delay prediction using classification"). The final project direction was my decision based on:
- Course requirements (ML component mandatory)
- Dataset availability (DataCo dataset on Kaggle)
- Personal interest in logistics optimization
- Feasibility within the project timeline

---

## Phase 2: Architecture and Design

### Tools Used
- **Claude (Anthropic)**

### How It Was Used

Once the project scope was defined, I used Claude for architectural discussions:

1. **Module structure**: Discussed how to organize the codebase into logical packages (`src/data`, `src/features`, `src/ml`) following separation of concerns.

2. **Design patterns**: Explored appropriate patterns for the pipeline architecture, ultimately choosing a class-based approach with clear interfaces between components.

3. **Configuration management**: Designed a centralized `config.py` approach to avoid hardcoded values and enable reproducibility.

### My Contributions
- Made final decisions on folder structure based on course guidelines
- Adapted suggestions to match my understanding of Python packaging
- Drew architecture diagrams independently based on my mental model

---

## Phase 3: Implementation

### Tools Used
- **Claude (Anthropic)**
- **GitHub Copilot**

### How They Were Used

#### Code Development with Claude
Claude assisted with:

1. **Boilerplate code**: Generated initial class structures with docstrings and type hints, which I then customized and extended.

2. **Debugging assistance**: When encountering errors, I described the problem and received explanations of potential causes. I then fixed the issues myself after understanding the root cause.

3. **Code review**: Submitted my code for review and received feedback on:
   - PEP 8 compliance
   - Potential edge cases
   - Testability improvements
   - Performance optimizations

4. **Feature engineering ideas**: Discussed domain-informed features (e.g., `shipping_mode_risk`, `market_risk_score`) and how to implement them. The specific risk values were based on my analysis of the dataset patterns.

5. **Test generation**: Received suggestions for test cases, which I adapted to match my specific implementation details.

#### Code Completion with GitHub Copilot
- Used for routine code completion (import statements, common patterns)
- All suggestions were reviewed before acceptance
- Rejected suggestions that didn't match my coding style or project needs

### Critical Learning Moment: Data Leakage

A significant learning experience occurred when initial models achieved ~95% accuracy. Rather than accepting this result, I:

1. Investigated which features drove predictions
2. Identified that `Days for shipping (real)` contained post-delivery information
3. Researched data leakage concepts in ML literature
4. Implemented a solution by adding leaky columns to the configuration blacklist
5. Re-ran experiments to get honest results (~69% accuracy)

This discovery came from my own critical analysis, not from AI suggestion. Claude helped explain the concept of data leakage when I asked, but identifying the specific leaky columns in my dataset was my own investigation.

---

## Phase 4: Testing

### Tools Used
- **Claude (Anthropic)**

### How It Was Used

1. **Test strategy**: Discussed testing approaches (unit tests, integration tests, fixtures) and received guidance on pytest best practices.

2. **Test case suggestions**: For each module, received suggestions for what to test (happy path, edge cases, error handling).

3. **Coverage improvement**: After initial tests, asked for suggestions on how to increase coverage for specific modules.

### My Contributions
- Wrote all test assertions based on my understanding of expected behavior
- Created test fixtures with realistic data matching the DataCo schema
- Debugged failing tests independently
- Achieved 80% coverage through iterative improvement

---

## Phase 5: Documentation and Report

### Tools Used
- **ChatGPT (OpenAI)**
- **Claude (Anthropic)**
- **Perplexity AI**

### How They Were Used

1. **Report structure**: Discussed how to organize the technical report according to IEEE format and course requirements.

2. **Writing refinement**: Submitted draft sections for feedback on clarity, technical accuracy, and academic tone.

3. **LaTeX formatting**: Received help with LaTeX syntax for tables, equations, and figures.

4. **Literature search**: Used Perplexity to find relevant academic references for the Background section.

### My Contributions
- All technical content based on my actual implementation and results
- Verified every number in the report against Nuvolos outputs
- Wrote the data leakage narrative from my own experience
- Created all figures programmatically from my code
- Made editorial decisions on what to include/exclude

---

## Additional Learning Resources

Beyond AI tools, I consulted the following resources:

### Books
- James, G. et al. (2021). *An Introduction to Statistical Learning* - For classification theory and evaluation metrics
- Guttag, J.V. (2021). *Introduction to Computation and Programming Using Python* - For software engineering practices

### Online Courses and Videos
- YouTube tutorials on scikit-learn pipeline design
- YouTube videos on pytest best practices
- Course lecture materials on pandas and data manipulation

### Documentation
- scikit-learn official documentation
- XGBoost documentation
- pandas documentation
- pytest documentation

### Stack Overflow
- Consulted for specific error messages and edge cases
- All solutions were understood before implementation

---

## Commitment to Academic Integrity

I affirm that:

1. **Understanding**: I can explain every design decision, algorithm choice, and line of code in this project.

2. **Original work**: While AI tools assisted with code generation and writing, all final decisions and implementations reflect my own understanding.

3. **Verification**: All results and metrics in the report were verified against actual outputs from running the code.

4. **Learning**: AI tools accelerated my learning but did not replace it. I invested significant time understanding concepts before and after receiving AI assistance.

5. **Transparency**: This document honestly represents how AI tools were used throughout the project.

---

## Summary Table

| Phase | Tools Used | Primary Purpose |
|-------|------------|-----------------|
| Ideation | ChatGPT, Perplexity | Brainstorming, research |
| Design | Claude | Architecture discussions |
| Implementation | Claude, Copilot | Code assistance, debugging |
| Testing | Claude | Test strategy, coverage |
| Documentation | ChatGPT, Claude, Perplexity | Writing support, references |

---

## Conclusion

AI tools were valuable collaborators in this project, helping me work more efficiently and learn new concepts faster. However, the intellectual work—understanding the problem, making design decisions, debugging issues, and validating results—remained my responsibility. This project represents genuine learning and skill development, enhanced but not replaced by AI assistance.

---

*This declaration is submitted in compliance with the course AI Tools Policy.*
