# Company HTML Structure (Step 2 findings)

File: `data/raw/company.html` | Document version: **1.0** | Effective: 1 April 2026

This document records what we learned by inspecting the knowledge source **before** writing the
parser. Re-generate the numbers any time with `python scripts/inspect_html.py`.

## 1. Size and shape

| Measure | Value |
|---|---|
| File size | about 33,000 characters |
| Useful words (after removing noise) | about 3,300 |
| Noise words | about 160 |
| Sections (`<h2>`) | 14 |
| Subsections (`<h3>`) | 46 |
| Tables | 11 |
| Lists | 15 `<ul>` + 5 `<ol>`, nested up to 2 levels |
| Definition list (`<dl>`) | 1 (the FAQ) |

## 2. Skeleton

```
<html>
  <head>      meta tags (version, owner, date), <title>, <script>, <style>
  <body>
    <header> + <nav>            <- NOISE
    <div class="cookie-banner"> <- NOISE
    <div style="display:none">  <- NOISE (hidden draft note)
    <main>
      <h1>                      document title
      <p class="doc-meta">      version / owner line
      <nav class="toc">         <- NOISE (duplicates headings)
      <section id="...">        x 14, each starts with exactly one <h2>
         <h3> subsection
         <p> / <ul> / <ol> / <table> / <div class="note">
      </section>
    </main>
    <footer>                    <- NOISE
    <noscript><img ...></noscript> <- NOISE (tracking pixel)
```

## 3. Useful elements and how the parser should treat them

| Element | Where used | Parser decision (Step 3) |
|---|---|---|
| `<h1>` | document title | Store as document title, not as a section. |
| `<h2>` | 14 main sections | Becomes the chunk's `section` metadata (e.g. "Leave Policy"). |
| `<h3>` | subsections | Becomes the chunk's `subsection` metadata (e.g. "Carry Forward and Encashment"). |
| `<p>` | normal text | Keep text; keep `<strong>` text, drop the tag. |
| `<ul>` / `<ol>` | lists, nested | One line per `<li>`; ordered lists keep numbering (steps matter); nested items stay with their parent. |
| `<table>` | 11 tables | Convert each row to a sentence that repeats the column names (see below). |
| `<dl>` | FAQ | Convert each `<dt>`/`<dd>` pair to "Question: ... Answer: ...". |
| `<div class="note">` | callouts | Keep as normal text (they hold important rules). |
| `<a>` | links | Keep the link text, drop the URL. |
| `<meta name="document-version">` | head | Read once, attach to every chunk as `document_version`. |

### Why tables need special handling

A table cell on its own has no meaning. The row `Casual Leave | 8 | Can be taken in half-day units...`
becomes useless if flattened to `Casual Leave 8 Can be taken...`. The parser should output:

> `Leave Type: Casual Leave; Days Per Year: 8; Rules: Can be taken in half-day units...`

Now the chunk contains the words a user would search for ("casual leave", "days per year").

## 4. Unwanted elements (must be removed)

| Element | How to recognise it |
|---|---|
| Tracking script and its loader | `<script>` (2 of them) |
| CSS rules | `<style>` |
| Site header and menu | `<header>`, `<nav class="main-nav">` |
| Table of contents | `<nav class="toc">` - repeats every heading, would pollute search results |
| Cookie banner | class `cookie-banner` |
| Hidden draft note | `style="display:none"` - **not official policy**, must never be answered from |
| Footer | `<footer>` (copyright text) |
| Tracking pixel | `<noscript>` |
| HTML comments | `<!-- ... -->` (22 of them) |

Rule of thumb for the parser: remove by tag (`script, style, nav, header, footer, noscript`),
then remove anything hidden, then remove by class keyword (`cookie`, `banner`, ...).

## 5. Section outline

| Section (`<h2>`) | id | Subsections (`<h3>`) |
|---|---|---|
| Company Overview | company-overview | About; Services; Mission and Vision; Core Values |
| Departments | departments | (table only) |
| Job Roles and Career Levels | job-levels | Career Levels; Role Responsibilities; Internal Transfers |
| Working Hours and Attendance | working-hours | Standard Working Hours; Core Hours and Flexible Start; Attendance Recording; Overtime and Weekend Work |
| Work Model | work-model | Hybrid Work Policy; Work From Anywhere; Remote Work Support |
| Leave Policy | leave-policy | Annual Paid Leave; Carry Forward and Encashment; Public Holidays; Special Leave; Leave Approval |
| Salary and Compensation | salary | Pay Cycle; Salary Structure; Salary Bands by Level; Deductions; Annual Performance Review and Increments |
| Benefits | benefits | Health and Insurance; Retirement Benefits; Learning and Development; Wellness and Work-Life Support; Recognition; Employee Referral Programme |
| Employee Policies | employee-policies | Code of Conduct; POSH; Dress Code; Confidentiality and Information Security; Outside Work and Conflict of Interest; Social Media; Travel and Expense Policy |
| HR Information | hr-information | Onboarding; Probation and Confirmation; Notice Period; Promotions; Grievance Redressal |
| Internal Processes | internal-processes | How to Apply for Leave; How to Claim an Expense Reimbursement; IT Support and Equipment; How to Refer a Candidate |
| Company Rules | company-rules | (list only) |
| Frequently Asked Questions | faq | (definition list) |
| Contact Directory | contacts | (table only) |

## 6. Test facts for later steps

These are deliberately placed in the document so that retrieval and answer tests have known targets.

**Present (the bot SHOULD answer):**

| Question | Expected section | Expected answer |
|---|---|---|
| How many paid leaves do employees get? | Leave Policy | 24 days per year |
| What are the working hours? | Working Hours and Attendance | 9:00 AM to 6:00 PM, Mon-Fri |
| When is salary paid? | Salary and Compensation | Last working day of the month |
| How long is the notice period after confirmation? | HR Information | 60 days (L1-L4), 90 days (L5+) |
| How much is the health insurance cover? | Benefits | Rs 5,00,000 |
| Which days are office days? | Work Model | Tuesday, Wednesday, Thursday |
| How soon must I report a lost laptop? | Employee Policies | Within 1 hour |

**Absent (the bot MUST refuse):** company revenue, profit, stock or share price, company valuation,
ESOPs / stock options, sabbatical leave, dental insurance.

**Traps (related but not stated):**
- A *sports club membership* appears only in a hidden draft note, so it must NOT be answered.
- Salary of a *specific named person* is never given (and the document says it is confidential).
- Offices in cities other than Bengaluru, Hyderabad and Chennai are not mentioned.

## 7. Measured block sizes (important for Step 4)

Measured by splitting the main content at every `<h2>`/`<h3>`:

| Measure | Value |
|---|---|
| Blocks (heading + its text) | 60 |
| Median length | about 45 words |
| Shortest | 1 word (an `<h2>` that is followed straight away by an `<h3>`) |
| Longest | about 258 words |
| Blocks under 30 words | 12 |
| Blocks over 200 words | 1 |

What this means for chunking:
- Most subsections are short, so **one subsection = one chunk** is a sensible starting point.
- Very short blocks (for example "Leave Approval", 2 sentences) may be better **merged** with their
  neighbours in the same section, otherwise the chunk carries too little context.
- Our document is small (about 3,300 words), so the doc's suggestion of 500-1000 tokens per chunk
  is an upper limit here, not a target. We will decide the real values by experiment in Step 4 and
  Step 18.

### Decision made in Step 4

We looked at the real chunks and decided:
- **Do not merge short subsections.** A short chunk with its heading (for example
  `Employee Policies > Dress Code`, 26 words) is still easy to retrieve and gives a precise citation.
  Merging would mix topics. We will re-check this with retrieval tests in Step 9.
- **Limit = 150 words** (about 200 tokens). With the old value of 800 nothing would be split, and
  the FAQ (about 280 words) and the Departments table (about 260 words) would become single, mixed-topic chunks.
  A limit around 150 words also stays under the 256-token input limit of many small embedding
  models (to be confirmed when we pick the model in Step 5).
- **Overlap = 20 words, used only when one block must be split inside.**
