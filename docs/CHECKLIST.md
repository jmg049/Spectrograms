Documentation Maintenance Checklist
===================================

When Adding New Features
-------------------------

- [ ] Update type stubs in ``python/spectrograms/__init__.pyi``
- [ ] Add docstrings to new Python functions/classes
- [ ] Add to appropriate RST file in ``docs/source/api/``
- [ ] Add usage example to relevant guide
- [ ] Update ``__all__`` in ``__init__.py``
- [ ] Build docs and check for warnings
- [ ] Add example file if feature is substantial

When Changing API
-----------------

- [ ] Update affected RST files
- [ ] Update code examples
- [ ] Check cross-references still work
- [ ] Update parameter descriptions
- [ ] Verify autodoc still picks up changes
- [ ] Update version number if needed

When Updating Documentation
---------------------------

- [ ] Follow the existing style (concise, educational)
- [ ] Include runnable code examples
- [ ] Add cross-references with ``:class:``, ``:func:``, etc.
- [ ] Test all code examples
- [ ] Build locally: ``python build_docs.py --serve``
- [ ] Check mobile responsiveness
- [ ] Verify search works

Before Release
--------------

- [ ] Clean build: ``python build_docs.py --clean``
- [ ] Check all links work
- [ ] Verify code examples are up to date
- [ ] Review API reference completeness
- [ ] Check for Sphinx warnings
- [ ] Test installation instructions
- [ ] Update version numbers
- [ ] Generate deployment zip: ``./deploy_docs.sh``

Quality Checks
--------------

**Completeness:**
- All public API elements documented?
- All parameters explained?
- Return values documented?
- Exceptions listed?

**Clarity:**
- Examples clear and focused?
- Technical terms explained?
- Common use cases covered?
- Appropriate cross-references?

**Accuracy:**
- Code examples tested?
- Parameter types correct?
- Default values accurate?
- Links not broken?

Common Issues
-------------

**Autodoc not finding module:**
- Check ``sys.path`` in ``conf.py``
- Verify module imports successfully
- Check for import errors in Python package

**Missing documentation:**
- Ensure docstrings present
- Check autodoc options
- Verify RST files reference correct paths

**Build warnings:**
- Fix reference targets
- Update toctree entries
- Check for syntax errors in RST files
