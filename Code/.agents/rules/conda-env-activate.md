---
trigger: glob
applyTo: 'GLP/**'
description: Activate 'glp' conda environment for any code execution within the GLP/ directory.
---

Whenever you execute code or run commands within the `GLP/` directory, you MUST activate the `glp` conda environment first. 

This rule applies exclusively to the `GLP/` scope. Ensure that any terminal command (like `python`, `pip`, etc.) is preceded by the activation command in the same execution block or terminal session.

**Activation Command:**
`conda activate glp`
