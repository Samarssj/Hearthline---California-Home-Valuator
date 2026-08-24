"""Smoke-test the Streamlit UI and theme/prediction state behavior."""

from streamlit.testing.v1 import AppTest


app = AppTest.from_file("app.py", default_timeout=30).run()
assert not app.exception, app.exception

button_labels = [button.label for button in app.button]
assert "☀  Light mode" in button_labels
assert "◐  Dark mode" in button_labels
assert "Estimate home value  →" in button_labels

# Submit the default property profile and confirm a result is rendered.
app.button[button_labels.index("Estimate home value  →")].click().run()
assert not app.exception, app.exception
assert len(app.markdown) > 0
assert app.session_state["prediction_result"] is not None
assert len(app.get("vega_lite_chart")) >= 2
assert len(app.session_state["prediction_comparison"]) == 4
prediction_before_theme_switch = app.session_state["prediction_result"]

# Click Dark mode and confirm the same result survives the Streamlit rerun.
button_labels = [button.label for button in app.button]
app.button[button_labels.index("◐  Dark mode")].click().run()
assert not app.exception, app.exception
assert app.session_state["theme_mode"] == "Dark"
assert app.session_state["prediction_result"] == prediction_before_theme_switch

print("Streamlit UI validation passed.")
print(f"Theme controls: {button_labels[:2]}")
print(f"Prediction retained after dark mode: ${prediction_before_theme_switch:,.2f}")
