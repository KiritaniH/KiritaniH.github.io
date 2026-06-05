---
name: scientific_report_formatter
description: Format a scientific report into a structured report section and extract important claims without judging their correctness.
---

# Report Writer

Use this skill when the user asks the agent to generate, rewrite, standardize, or format a scientific report based on provided research materials, experiment results, paper summaries, tool outputs, or previous conversation content.

This skill does not evaluate whether the claims are correct.  
This skill only standardizes the report structure and extracts important claims from the generated report.

## Purpose

The purpose of this skill is to make the agent's scientific report output more structured, readable, and easier to evaluate later.

The output must contain two main parts:

1. A normal scientific report.
2. A list of important claims extracted from the report.

The second part only lists important claims.  
Do not judge whether the claims are supported, partially supported, unsupported, or contradicted.  
Do not classify errors.  
Do not output a claim-evidence consistency table.

## Input

The user may provide one or more of the following:

- Research topic
- Paper abstract
- Paper excerpt
- Experiment results
- Tables
- Model outputs
- Tool outputs
- Previous conversation content
- Requirements for report type, such as related work, method summary, experiment analysis, limitation analysis, or discussion

Use only the information provided by the user or by tools in the current conversation unless the user explicitly asks for external retrieval.

## Output Format

Always output in the following structure:

# [Report Title]

## Part I. Scientific Report

### 1. Research Background

Briefly introduce the research problem, task setting, or motivation.

If the input material does not contain enough background information, keep this section concise and do not invent unsupported background details.

### 2. Input Material Analysis

Summarize the key information from the provided materials.

This section may include:

- Important definitions
- Experimental settings
- Data sources
- Model or method descriptions
- Evaluation metrics
- Main numerical results
- Important limitations explicitly stated in the input

### 3. Logical Reasoning

Explain how the main conclusions are derived from the input materials.

Distinguish between:

- Directly stated facts
- Comparisons based on provided numbers
- Reasonable inferences from the given material
- Uncertain points that require further evidence

Do not present uncertain inferences as proven facts.

### 4. Conclusion

Give a concise conclusion based on the input materials and the reasoning above.

If the evidence is insufficient for a strong conclusion, explicitly state the limitation.

### 5. Summary

Provide a short summary of the report in 3-5 sentences.

## Part II. Important Claim Statements

Extract several important claims from Part I.

Output them as a numbered list.

Each claim should be:

- A complete sentence
- Specific enough to be checked later
- Directly related to the report
- Not too vague
- Not merely a section heading
- Not a duplicate of another claim

Use the following format:

1. [Claim 1]
2. [Claim 2]
3. [Claim 3]
4. [Claim 4]
5. [Claim 5]

The number of claims should usually be between 5 and 8 unless the user specifies otherwise.

## Important Rules

1. Do not evaluate the claims in Part II.
2. Do not output Judgment.
3. Do not output Error Type.
4. Do not classify claims as Supported, Partially Supported, Unsupported, or Contradiction.
5. Do not create a claim-evidence consistency table.
6. Do not add citations, evidence IDs, or source labels unless they already appear in the input or the user asks for them.
7. Do not invent paper titles, experimental results, numerical values, citations, or source materials.
8. If the input material is insufficient, state the insufficiency in the report instead of filling the gap with unsupported content.
9. Keep the report concise, formal, and suitable for a scientific project report.
10. Preserve important numerical values and experimental settings exactly when they are provided.

## Example Output Template

# Report Title

## Part I. Scientific Report

### 1. Research Background

[Write the background here.]

### 2. Input Material Analysis

[Analyze the provided materials here.]

### 3. Logical Reasoning

[Explain the reasoning process from materials to conclusions here.]

### 4. Conclusion

[Write the main conclusion here.]

### 5. Summary

[Write a short summary here.]

## Part II. Important Claim Statements

1. [Important claim 1.]
2. [Important claim 2.]
3. [Important claim 3.]
4. [Important claim 4.]
5. [Important claim 5.]
