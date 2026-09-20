"""Shared plumbing for the OptionLab Streamlit app.

``state``  — session state, the sidebar market controls, the book and the simulator.
``ui``     — the small component library every page is built from (headers, charts,
             metric rows, explanation boxes, parameter controls, formatting).

Pages live in ``app_pages/`` and import from here; they never talk to
``st.session_state`` keys directly.
"""

__all__ = ["state", "ui"]
