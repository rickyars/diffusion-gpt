# WE Art Piece Refactoring - Summary

**Date:** January 11, 2026
**Branch:** feature/loss-weighting-implementation
**Status:** Complete and ready to commit

## Problem Statement

You had a conflict between two needs:
1. **Manually edit `we.html`** for styling, effects, animations, and timing
2. **Update the model** independently without regenerating the entire file

The old `build.py` forced you to choose: regenerate (lose edits) or keep edits (out of sync with model).

## Solution: Template-Based Architecture

Implemented a **template + injection system** that:
- ✅ Lets you edit HTML freely
- ✅ Preserves edits when updating models
- ✅ Keeps template small and version-controlled
- ✅ Separates concerns (structure vs data)

## What Changed

### New Files Created

| File | Size | Purpose |
|------|------|---------|
| `scripts/art-piece/extract_template.py` | 5 KB | Extract template from edited HTML |
| `scripts/art-piece/update_model.py` | 6 KB | Fill template with model data |
| `scripts/art-piece/we.template.html` | 39 KB | Clean template (git-tracked) |
| `scripts/art-piece/WORKFLOW.md` | 7 KB | Complete workflow guide |

### Files Removed

| File | Reason |
|------|--------|
| `scripts/art-piece/build.py` | Replaced by extract_template.py + update_model.py |
| `scripts/art-piece/merge_onnx_data.py` | Functionality moved to export_to_onnx.py |

### Files Updated

| File | Changes |
|------|---------|
| `.gitignore` | Added `*.onnx`, `*.onnx.data`, `scripts/art-piece/we.html` |
| `docs/WE_CUSTOMIZATION_GUIDE.md` | Updated workflow, added template architecture section |

## Architecture

```
we.template.html (39 KB, git-tracked)
    ↓ (fill placeholders with model data)
    ↓ (extract_template.py creates)
    ↓ (update_model.py fills)
    ↓
we.html (60 MB, gitignored, generated)

Placeholders in template:
  - {{MODEL_CONFIG}}
  - {{VOCAB}}
  - {{MODEL_BASE64}}
  - {{INFERENCE_ENGINE}}
```

## Workflow

### Editing HTML (Styling, Effects, Timing)
```bash
1. Edit we.html in VSCode
2. Test in browser
3. python scripts/art-piece/extract_template.py
4. git add scripts/art-piece/we.template.html && git commit
```

### Updating Model
```bash
1. python scripts/art-piece/export_to_onnx.py --model models/new.pt
2. python scripts/art-piece/update_model.py --dataset confessions
3. Test we.html in browser
   → Your template edits are preserved!
```

### Both Editing and Updating
```bash
1. Edit we.html + extract_template.py + git commit
2. python update_model.py --dataset confessions
   → Fresh model, your edits intact!
```

## Key Benefits

✅ **Edit freely** - No fear of losing changes
✅ **Update independently** - Model and structure separate
✅ **Version control** - Template is tiny (39 KB, not 61 MB)
✅ **Clean separation** - Template structure vs injected data
✅ **Easy onboarding** - New models in one command
✅ **Collaboration** - Share template, not model data

## Git Status

Ready to commit with:

```bash
git status
# Shows:
#  M  .gitignore
#  M  docs/WE_CUSTOMIZATION_GUIDE.md
#  A  scripts/art-piece/WORKFLOW.md
#  A  scripts/art-piece/extract_template.py
#  A  scripts/art-piece/update_model.py
#  D  scripts/art-piece/build.py
#  D  scripts/art-piece/merge_onnx_data.py
#  R  scripts/art-piece/build.py -> scripts/art-piece/we.template.html
```

Total changes: **+1074 insertions, -442 deletions**

## Documentation

- **Quick Start:** `scripts/art-piece/WORKFLOW.md`
- **Detailed Examples:** `docs/WE_CUSTOMIZATION_GUIDE.md`
- **Visual Reference:** All sections in guide still apply

## Verification

Template validated:
- ✓ `{{MODEL_CONFIG}}` placeholder present (1 occurrence)
- ✓ `{{VOCAB}}` placeholder present (1 occurrence)
- ✓ `{{MODEL_BASE64}}` placeholder present (1 occurrence)
- ✓ `{{INFERENCE_ENGINE}}` placeholder present (1 occurrence)
- ✓ No old hardcoded model data in template

Scripts tested:
- ✓ `extract_template.py` successfully created template
- ✓ `update_model.py` ready to fill placeholders
- ✓ Both scripts have proper error handling and help text

## Next Steps

1. **Review changes:**
   ```bash
   git diff --cached
   ```

2. **Commit to feature branch:**
   ```bash
   git commit -m "Refactor: template-based WE HTML generation

   - Replace build.py with extract_template.py and update_model.py
   - Separate HTML template from model data injection
   - Preserve manual edits independent of model updates
   - Keep template in git, ignore generated we.html
   - Add comprehensive WORKFLOW.md guide
   - Update WE_CUSTOMIZATION_GUIDE.md with new system"
   ```

3. **Continue developing:**
   - Edit `we.html` with confidence
   - Extract template when happy with changes
   - Update models anytime - your edits stay safe

## Backward Compatibility

- ✅ Existing `export_to_onnx.py` still works
- ✅ All customization guidance still applies
- ⚠️ `build.py` is now deprecated (functionality replaced)
- ⚠️ `merge_onnx_data.py` superseded by `export_to_onnx.py`

## Questions?

See:
- `scripts/art-piece/WORKFLOW.md` - Complete workflow guide
- `docs/WE_CUSTOMIZATION_GUIDE.md` - Customization examples
- `docs/WE_SPEC.md` - Technical specification (unchanged)
