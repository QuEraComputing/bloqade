coverage-run:
    coverage run -m pytest test

coverage-xml:
    coverage xml

coverage-html:
    coverage html

coverage-report:
    coverage report

coverage-open:
    open htmlcov/index.html

coverage: coverage-run coverage-xml coverage-report

doc FLAGS="":
    mkdocs serve {{FLAGS}}

doc-build FLAGS="":
    mkdocs build {{FLAGS}}

# Parse-check the packaged Plotly interaction controller and run its tests. The
# ``{plot_id}`` token is inside a JavaScript string literal, so it is valid
# before Plotly substitutes the concrete div ID during HTML generation.
check-visualization-js:
    node --check src/bloqade/gemini/visualize/_arch_interactive.js
    node --test test/gemini/visualize/test_arch_interactive.cjs
