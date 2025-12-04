#!/usr/bin/env python
"""
Lightweight local web UI to explore CME loss behaviour.

Launch:
    python loss_lab_app.py --port 8000

Open http://localhost:8000, choose a mask type, target constraints, and a loss
variant, then the page renders loss value, gradient heatmap, gradient norm, and
angular profile vs target.
"""
import argparse
import base64
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import matplotlib

# Headless backend for server usage
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from networks.losses.cme_losses import CMELossAngularProfileMSE, CMELossAngularProfileWasserstein, build_target  # noqa: E402
from utils.sanity_checkers.sanity_check_gradient_loss import (  # noqa: E402
    build_constraints_set,
    build_fake_mask,
)


DEFAULT_H = 360
DEFAULT_W = 360
DEVICE = "cpu"


def build_manual_sector_mask(H, W, pa_deg, aw_deg, inner_frac, outer_frac, device):
    """
    Build a simple ring sector mask from PA/AW and radial fractions.
    """
    aw_deg = max(0.0, float(aw_deg))
    pa_deg = float(pa_deg) % 360
    inner_frac = float(inner_frac)
    outer_frac = float(outer_frac)
    if outer_frac <= inner_frac:
        outer_frac = min(1.0, inner_frac + 0.1)

    # Radial mask
    r = torch.linspace(0, 1, steps=H, device=device)
    radial_mask = (r >= inner_frac) & (r <= outer_frac)
    radial_mask = radial_mask[:, None]  # [H,1]

    # Angular mask
    theta = torch.linspace(0, 360, steps=W, device=device, dtype=torch.float32)
    amin = (pa_deg - aw_deg / 2) % 360
    amax = (pa_deg + aw_deg / 2) % 360
    if amin < amax:
        ang_mask = (theta >= amin) & (theta <= amax)
    else:
        ang_mask = (theta >= amin) | (theta <= amax)
    ang_mask = ang_mask[None, :]  # [1,W]

    m = (radial_mask & ang_mask).float()  # broadcast to [H,W]
    return m.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def make_loss(loss_name, form_values):
    """Instantiate the selected loss with optional parameters from the form."""
    if loss_name == "CMELossAngularProfileMSE":
        gaussian = form_values.get("gaussian_target", ["off"])[0] == "on"
        sigma = float(form_values.get("sigma_target", [10.0])[0])
        return CMELossAngularProfileMSE(gaussian_target=gaussian, sigma_target=sigma, n_bins=DEFAULT_W)
    if loss_name == "CMELossAngularProfileWasserstein":
        gaussian = form_values.get("gaussian_target", ["off"])[0] == "on"
        sigma = float(form_values.get("sigma_target", [10.0])[0])
        return CMELossAngularProfileWasserstein(gaussian_target=gaussian, sigma_target=sigma, n_bins=DEFAULT_W)
    raise ValueError(f"Unknown loss: {loss_name}")


def parse_constraints(form_values):
    """
    Build constraints list based on form selection.
    - preset: uses the same presets as the old sanity_check script.
    - manual: uses pa/da fields and derives theta_min/theta_max.
    """
    mode = form_values.get("constraint_mode", ["preset"])[0]
    if mode == "manual":
        try:
            pa = float(form_values.get("pa", [0])[0])
            da = float(form_values.get("da", [0])[0])
        except ValueError:
            return [[]]

        theta_min = (pa - da / 2) % 360
        theta_max = (pa + da / 2) % 360

        return [[{"theta_min": theta_min, "theta_max": theta_max, "pa": pa, "da": da}]]

    preset = form_values.get("constraint_preset", ["noCME"])[0]
    return build_constraints_set(preset)


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def compute_analysis(form_values):
    """Run forward/backward pass for the chosen configuration and render plots."""
    mask_type = form_values.get("mask_type", ["constant"])[0]
    loss_name = form_values.get("loss_name", ["CMELossAngularProfileMSE"])[0]

    constraints_batch = parse_constraints(form_values)
    loss_fn = make_loss(loss_name, form_values).to(DEVICE)

    if mask_type == "manual_sector":
        pa = float(form_values.get("mask_pa", [150])[0])
        aw = float(form_values.get("mask_aw", [40])[0])
        inner = float(form_values.get("mask_inner", [0.2])[0])
        outer = float(form_values.get("mask_outer", [0.8])[0])
        mask0 = build_manual_sector_mask(DEFAULT_H, DEFAULT_W, pa, aw, inner, outer, DEVICE)
    else:
        mask0 = build_fake_mask(mask_type, DEFAULT_H, DEFAULT_W, DEVICE)
    mask_pred = mask0.clone().detach().requires_grad_(True)

    # Placeholder input to match loss signature
    X = torch.zeros((1, 1, DEFAULT_H, DEFAULT_W), device=DEVICE)

    loss, _ = loss_fn(mask_pred, constraints_batch, X)
    loss.backward()

    grad = mask_pred.grad.detach().cpu().numpy()[0, 0]
    mask_np = mask_pred.detach().cpu().numpy()[0, 0]

    # Angular profiles
    A_pred = mask_np.mean(axis=0)
    gaussian = form_values.get("gaussian_target", ["off"])[0] == "on"
    sigma = float(form_values.get("sigma_target", [10.0])[0])
    target = build_target(DEFAULT_W, constraints_batch[0], device=torch.device("cpu"), gaussian=gaussian, sigma=sigma).numpy()

    # ----- Plot: gradient heatmap -----
    m = float(np.abs(grad).max()) + 1e-12
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    im1 = ax1.imshow(grad, cmap="bwr", norm=Normalize(vmin=-m, vmax=m))
    ax1.set_title("Gradient heatmap")
    fig1.colorbar(im1, ax=ax1, shrink=0.8)
    grad_heatmap = fig_to_base64(fig1)

    # ----- Plot: |gradient| -----
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im2 = ax2.imshow(np.abs(grad), cmap="viridis")
    ax2.set_title("|Gradient|")
    fig2.colorbar(im2, ax=ax2, shrink=0.8)
    grad_norm = fig_to_base64(fig2)

    # ----- Plot: angular profiles -----
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot(A_pred, label="Pred (angular mean)")
    ax3.plot(target, label="Target", linestyle="--")
    ax3.set_title("Angular profile: pred vs target")
    ax3.set_xlabel("Angle bin (deg)")
    ax3.grid(alpha=0.3)
    ax3.legend()
    prof_angle = fig_to_base64(fig3)

    # ----- Plot: input mask -----
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    im4 = ax4.imshow(mask_np, cmap="gray")
    ax4.set_title("Mask")
    fig4.colorbar(im4, ax=ax4, shrink=0.8)
    mask_img = fig_to_base64(fig4)

    return {
        "loss_value": float(loss.item()),
        "grad_heatmap": grad_heatmap,
        "grad_norm": grad_norm,
        "prof_angle": prof_angle,
        "mask_img": mask_img,
    }


def render_form(results=None):
    """Return the HTML page with form and optional results."""
    mask_options = [
        "manual_sector",
        "empty",
        "full_disk",
        "single_sector",
        "double_sector",
        "ring",
        "edge_flash",
        "noisy_cloud",
        "random",  # kept for experimentation
    ]
    constraint_options = [
        "noCME",
        "CME_standard",
        "CME_wrap",
        "CME_double",
        "CME_double2",
        "CME_double3",
    ]
    loss_options = [
        "CMELossAngularProfileMSE",
        "CMELossAngularProfileWasserstein",
    ]

    def select(options, current):
        return "".join(
            f'<option value="{o}" {"selected" if o == current else ""}>{o}</option>'
            for o in options
        )

    form_values = results.get("form_values") if results else {}

    def get_value(name, default):
        val = form_values.get(name, [default])[0]
        return val if val != "" else default

    # Current selections
    cur_mask = get_value("mask_type", "constant")
    cur_constraint_mode = get_value("constraint_mode", "preset")
    cur_constraint_preset = get_value("constraint_preset", "noCME")
    cur_loss = get_value("loss_name", "CMELossAngularProfileMSE")
    gaussian_checked = "checked" if form_values.get("gaussian_target", ["off"])[0] == "on" else ""
    pa_val = get_value("pa", "150")
    da_val = get_value("da", "60")
    sigma_target_val = get_value("sigma_target", "10")
    mask_pa_val = get_value("mask_pa", "150")
    mask_aw_val = get_value("mask_aw", "40")
    mask_inner_val = get_value("mask_inner", "0.2")
    mask_outer_val = get_value("mask_outer", "0.8")

    error_msg = ""
    if results and results.get("error"):
        error_msg = f"<p style='color:red;'>Error: {results['error']}</p>"

    result_block = ""
    if results and "loss_value" in results:
        result_block = f"""
        <div class="results">
            <h2>Results</h2>
            <p><strong>Loss:</strong> {results["loss_value"]:.6f}</p>
            <div class="img-row">
                <div><h3>Mask</h3><img src="data:image/png;base64,{results["mask_img"]}" /></div>
                <div><h3>Gradient heatmap</h3><img src="data:image/png;base64,{results["grad_heatmap"]}" /></div>
                <div><h3>|Gradient|</h3><img src="data:image/png;base64,{results["grad_norm"]}" /></div>
            </div>
            <div class="img-row">
                <div><h3>Angular profile</h3><img src="data:image/png;base64,{results["prof_angle"]}" /></div>
            </div>
        </div>
        """

    return f"""
    <html>
    <head>
        <title>CME Loss Lab</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
            fieldset {{ border: 1px solid #ccc; padding: 12px; }}
            legend {{ font-weight: bold; }}
            label {{ display: block; margin-top: 8px; }}
            input[type="number"] {{ width: 120px; }}
            .img-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .results {{ margin-top: 24px; }}
            button {{ padding: 10px 16px; font-size: 14px; }}
        </style>
    </head>
    <body>
        <h1>CME Loss Laboratory</h1>
        <form method="POST">
            <div class="grid">
                <fieldset>
                    <legend>Mask</legend>
                    <label>Preset:
                        <select name="mask_type">
                            {select(mask_options, cur_mask)}
                        </select>
                    </label>
                    <p><em>Manual sector (only if selected):</em></p>
                    <label>mask PA (deg): <input type="number" name="mask_pa" step="1" value="{mask_pa_val}"></label>
                    <label>mask AW (deg): <input type="number" name="mask_aw" step="1" value="{mask_aw_val}"></label>
                    <label>inner radius (0-1): <input type="number" name="mask_inner" step="0.05" value="{mask_inner_val}" min="0" max="1"></label>
                    <label>outer radius (0-1): <input type="number" name="mask_outer" step="0.05" value="{mask_outer_val}" min="0" max="1"></label>
                </fieldset>
                <fieldset>
                    <legend>Constraints / Target</legend>
                    <label><input type="radio" name="constraint_mode" value="preset" {"checked" if cur_constraint_mode == "preset" else ""}> Preset</label>
                    <label>Preset list:
                        <select name="constraint_preset">
                            {select(constraint_options, cur_constraint_preset)}
                        </select>
                    </label>
                    <label><input type="radio" name="constraint_mode" value="manual" {"checked" if cur_constraint_mode == "manual" else ""}> Manual (PA/AW)</label>
                    <label>pa (deg): <input type="number" name="pa" step="1" value="{pa_val}"></label>
                    <label>da (deg): <input type="number" name="da" step="1" value="{da_val}"></label>
                </fieldset>
                <fieldset>
                    <legend>Loss</legend>
                    <label>Loss class:
                        <select name="loss_name">
                            {select(loss_options, cur_loss)}
                        </select>
                    </label>
                    <label><input type="checkbox" name="gaussian_target" {gaussian_checked}> gaussian_target (MSE/Wasserstein)</label>
                    <label>sigma_target: <input type="number" name="sigma_target" step="1" value="{sigma_target_val}"></label>
                </fieldset>
            </div>
            <p><button type="submit">Run loss check</button></p>
        </form>
        {error_msg}
        {result_block}
    </body>
    </html>
    """


class LossLabHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = render_form().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form_values = parse_qs(body)

        try:
            results = compute_analysis(form_values)
            results["form_values"] = form_values
            html = render_form(results).encode("utf-8")
        except Exception as e:
            html = render_form({"form_values": form_values, "error": str(e)}).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


def main():
    parser = argparse.ArgumentParser(description="Start the CME loss lab server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the local server.")
    args = parser.parse_args()

    server = HTTPServer(("localhost", args.port), LossLabHandler)
    print(f"Loss lab running on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
def build_manual_sector_mask(H, W, pa_deg, aw_deg, inner_frac, outer_frac, device):
    """
    Simple interactive-friendly mask: a ring sector defined by PA/AW and radial bounds.
    """
    aw_deg = max(0.0, float(aw_deg))
    pa_deg = float(pa_deg) % 360
    inner_frac = float(inner_frac)
    outer_frac = float(outer_frac)
    if outer_frac <= inner_frac:
        outer_frac = min(1.0, inner_frac + 0.1)

    # Radial bounds in pixels
    r = torch.linspace(0, 1, H, device=device)
    radial_mask = (r >= inner_frac) & (r <= outer_frac)
    radial_mask = radial_mask[:, None]  # [H,1]

    # Angular mask
    theta = torch.linspace(0, 360, W, device=device, dtype=torch.float32, endpoint=False)
    amin = (pa_deg - aw_deg / 2) % 360
    amax = (pa_deg + aw_deg / 2) % 360
    if amin < amax:
        ang_mask = (theta >= amin) & (theta <= amax)
    else:
        ang_mask = (theta >= amin) | (theta <= amax)
    ang_mask = ang_mask[None, :]  # [1,W]

    m = (radial_mask & ang_mask).float()  # broadcast to [H,W]
    return m.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
