import reflex as rx


config = rx.Config(
    app_name="deformablebench",
    app_module_import="deformablebench.app",
    frontend_port=3000,
    backend_port=8000,
    frontend_packages=["three@0.166.1"],
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.RadixThemesPlugin(),
    ],
)
