import reflex as rx

from fabricbench.app.newton_bridge import add_newton_routes
from fabricbench.app.three_viewport import three_viewport


def index() -> rx.Component:
    """A full-page Reflex viewport with an interactive Three.js scene."""

    return rx.box(
        three_viewport(id="three-viewport"),
        width="100vw",
        height="100vh",
        margin="0",
        overflow="hidden",
    )


app = rx.App()
add_newton_routes(app)
app.add_page(index, route="/")
